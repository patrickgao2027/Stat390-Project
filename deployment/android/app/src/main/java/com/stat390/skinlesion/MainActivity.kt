package com.stat390.skinlesion

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.stat390.skinlesion.databinding.ActivityMainBinding
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Single-activity skin-lesion screener.
 *
 * Layout:  CameraX PreviewView fills most of screen, "Capture & Analyze" button
 * at the bottom, result TextView and disclaimer below the button.
 *
 * Flow:    Request CAMERA permission → bind CameraX preview + ImageCapture →
 *          on button tap, capture JPEG → decode to Bitmap → run [SkinLesionClassifier]
 *          on background thread → update result text on main thread.
 *
 * The classifier and its TFLite interpreter are created once in onCreate
 * (loading the model takes ~200 ms) and closed in onDestroy.
 */
class MainActivity : AppCompatActivity() {

    companion object { private const val TAG = "MainActivity" }

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: SkinLesionClassifier
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = SkinLesionClassifier(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        binding.captureButton.setOnClickListener { captureAndAnalyze() }
        binding.resultText.text = getString(R.string.result_placeholder)
        binding.disclaimerText.text = getString(R.string.disclaimer)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) startCamera()
        else cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()

            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) {
                Log.e(TAG, "CameraX bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndAnalyze() {
        val capture = imageCapture ?: return
        binding.captureButton.isEnabled = false
        binding.resultText.text = getString(R.string.result_analyzing)

        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: androidx.camera.core.ImageProxy) {
                val bitmap = imageProxyToBitmap(image)
                image.close()

                CoroutineScope(Dispatchers.Default).launch {
                    val t0 = System.currentTimeMillis()
                    val result = classifier.classify(bitmap)
                    val ms = System.currentTimeMillis() - t0
                    withContext(Dispatchers.Main) {
                        binding.resultText.text =
                            "${result.summaryString()}\n(${ms} ms, TTA=${result.usedTta})"
                        binding.captureButton.isEnabled = true
                    }
                }
            }

            override fun onError(exc: ImageCaptureException) {
                Log.e(TAG, "Capture error", exc)
                runOnUiThread {
                    binding.resultText.text = getString(R.string.result_error)
                    binding.captureButton.isEnabled = true
                }
            }
        })
    }

    /** Decode the CameraX ImageProxy (JPEG) into a Bitmap. */
    private fun imageProxyToBitmap(image: androidx.camera.core.ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining()).also { buffer.get(it) }
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.close()
    }
}

package com.stat390.skinlesion

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max

/**
 * TFLite wrapper for the skin-lesion classifier.
 *
 * Loads skin_lesion_b4.tflite from assets/, picks the fastest available delegate
 * (NNAPI > GPU > CPU), and exposes [classify] which:
 *  1. resizes the Bitmap to 128x128 (matches the frozen training pipeline)
 *  2. normalizes to float32 in [0, 1] NHWC
 *  3. optionally runs 4-view TTA (original + hflip + vflip + both) and averages
 *  4. compares averaged probability to [THRESHOLD] to produce the binary label
 *
 * THRESHOLD comes from convert.py output (printed when checkpoint is converted).
 */
class SkinLesionClassifier(context: Context) {

    companion object {
        private const val TAG = "SkinLesionClassifier"
        private const val MODEL_ASSET = "skin_lesion_b4.tflite"
        private const val INPUT_SIZE = 128
        private const val NUM_BYTES_PER_FLOAT = 4

        // From convert.py output line:
        //   "Use threshold = 0.0500 in Android app"
        // Update this constant after each new training/conversion cycle.
        private const val THRESHOLD = 0.05f

        // Toggle 4-view TTA. Slower (~3-5 sec) but ~0.02 better recall.
        // Set false for fast mode (~0.5-1 sec) at small accuracy cost.
        private const val USE_TTA = true
    }

    private val interpreter: Interpreter
    private val delegate: AutoCloseable?
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(INPUT_SIZE * INPUT_SIZE * 3 * NUM_BYTES_PER_FLOAT)
        .order(ByteOrder.nativeOrder())
    private val outputBuffer: Array<FloatArray> = arrayOf(FloatArray(1))

    init {
        val model = loadModelFile(context)
        val options = Interpreter.Options()
        val compat = CompatibilityList()

        delegate = when {
            tryAddNnapi(options) -> { Log.i(TAG, "Using NNAPI delegate"); null /* added directly */ }
            compat.isDelegateSupportedOnThisDevice -> {
                Log.i(TAG, "Using GPU delegate")
                val d = GpuDelegate(compat.bestOptionsForThisDevice)
                options.addDelegate(d); d
            }
            else -> { Log.i(TAG, "Using CPU (4 threads)"); options.setNumThreads(4); null }
        }
        interpreter = Interpreter(model, options)
    }

    /** Try to add NNAPI delegate; return true if successful. */
    private fun tryAddNnapi(opts: Interpreter.Options): Boolean = try {
        opts.addDelegate(NnApiDelegate()); true
    } catch (e: Throwable) {
        Log.w(TAG, "NNAPI unavailable: ${e.message}"); false
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_ASSET)
        FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    /**
     * Run inference on one Bitmap.
     *
     * @return [PredictionResult] with probability, binary label, and per-view probs (if TTA on)
     */
    fun classify(bitmap: Bitmap): PredictionResult {
        val resized = if (bitmap.width != INPUT_SIZE || bitmap.height != INPUT_SIZE) {
            Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        } else bitmap

        val probs: FloatArray = if (USE_TTA) {
            val views = arrayOf(
                resized,
                flip(resized, horizontal = true),
                flip(resized, horizontal = false),
                flip(flip(resized, horizontal = true), horizontal = false),
            )
            FloatArray(views.size) { i -> runOne(views[i]) }
        } else {
            floatArrayOf(runOne(resized))
        }

        val avgProb = probs.average().toFloat()
        val malignant = avgProb >= THRESHOLD
        return PredictionResult(
            probability = avgProb,
            isMalignant = malignant,
            perViewProbs = probs,
            usedTta = USE_TTA,
        )
    }

    /** Single forward pass: writes [bitmap] to inputBuffer, runs interpreter, returns prob. */
    private fun runOne(bitmap: Bitmap): Float {
        bitmapToInputBuffer(bitmap)
        interpreter.run(inputBuffer, outputBuffer)
        return outputBuffer[0][0]
    }

    /** Normalize Bitmap pixels to float32 [0, 1] in NHWC (interleaved) order.
     *  Matches the TFLite model's input shape (1, 128, 128, 3) — the sanity
     *  check in deployment/convert.py confirmed PyTorch and TFLite produce
     *  identical outputs when fed this layout. */
    private fun bitmapToInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (px in pixels) {
            inputBuffer.putFloat(((px shr 16) and 0xff) / 255f)   // R
            inputBuffer.putFloat(((px shr 8)  and 0xff) / 255f)   // G
            inputBuffer.putFloat((px          and 0xff) / 255f)   // B
        }
        inputBuffer.rewind()
    }

    private fun flip(bitmap: Bitmap, horizontal: Boolean): Bitmap {
        val matrix = Matrix().apply {
            if (horizontal) preScale(-1f, 1f) else preScale(1f, -1f)
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, false)
    }

    fun close() {
        try { interpreter.close() } catch (_: Throwable) {}
        try { (delegate as? AutoCloseable)?.close() } catch (_: Throwable) {}
    }
}

data class PredictionResult(
    val probability: Float,         // averaged P(malignant) across views
    val isMalignant: Boolean,        // probability >= THRESHOLD
    val perViewProbs: FloatArray,    // raw per-view probabilities (1 entry if TTA off)
    val usedTta: Boolean,
) {
    fun summaryString(): String {
        val pct = (probability * 100).toInt()
        return if (isMalignant)
            "⚠ Likely malignant (P=$pct%) — recommend dermatologist consultation"
        else
            "Likely benign (P=$pct%)"
    }
}

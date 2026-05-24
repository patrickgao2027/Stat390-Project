plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.stat390.skinlesion"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.stat390.skinlesion"
        minSdk = 26                  // covers all S22+ devices
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false  // keep TFLite delegates loadable
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    buildFeatures { viewBinding = true }

    // Don't compress the .tflite asset — TFLite needs it as-is for mmap loading.
    androidResources { noCompress += "tflite" }
}

dependencies {
    // Kotlin + AndroidX core
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")

    // CameraX — modern camera API
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // TensorFlow Lite — using 2.13.0 (last pre-split version) for AGP 9.x
    // compatibility. NNAPI delegate (built into core tensorflow-lite) routes
    // to the S22+ Snapdragon NPU, which is faster than the GPU delegate
    // anyway — so we drop the GPU delegate complexity entirely.
    implementation("org.tensorflow:tensorflow-lite:2.13.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
}

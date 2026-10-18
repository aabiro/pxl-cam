import java.io.File
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.licensee)
    alias(libs.plugins.triplet.play)
}

licensee {
    allow("Apache-2.0")
    allow("MIT")
    ignoreDependencies("com.google.android.gms") {
        transitive = true
        because("Android SDK License Agreement (proprietary); listed separately in-app")
    }
    ignoreDependencies("com.google.android.play") {
        transitive = true
        because("Android SDK License Agreement (proprietary); listed separately in-app")
    }
    bundleAndroidAsset = true
}

val playCredentialsPath =
    providers
        .gradleProperty("pxlcam.play.credentials")
        .orElse(providers.environmentVariable("PXLCAM_PLAY_CREDENTIALS"))
        .map { it.trim() }
        .filter { it.isNotEmpty() }

play {
    track.set("internal")
    defaultToAppBundles.set(true)
    enabled.set(false)
    if (playCredentialsPath.isPresent) {
        serviceAccountCredentials.set(File(playCredentialsPath.get()))
    }
}

fun parseSigningEnv(text: String): Map<String, String> =
    text
        .lineSequence()
        .map { it.trim() }
        .filter { it.isNotEmpty() && !it.startsWith("#") }
        .mapNotNull { line ->
            val body = line.removePrefix("export ").trim()
            val separator = body.indexOf('=')
            if (separator <= 0) {
                null
            } else {
                val key = body.substring(0, separator).trim()
                val rawValue = body.substring(separator + 1).trim()
                val value =
                    when {
                        rawValue.length >= 2 && rawValue.startsWith("\"") && rawValue.endsWith("\"") ->
                            rawValue.substring(1, rawValue.length - 1)
                        rawValue.length >= 2 && rawValue.startsWith("'") && rawValue.endsWith("'") ->
                            rawValue.substring(1, rawValue.length - 1)
                        else -> rawValue
                    }
                key to value
            }
        }.toMap()

fun Project.signingEnvFile(relativePath: String): Map<String, String> {
    val text =
        providers
            .fileContents(rootProject.layout.projectDirectory.file(relativePath))
            .asText
            .orNull
            ?: return emptyMap()
    return parseSigningEnv(text)
}

val localSigningEnv = signingEnvFile("release-signing.env") + signingEnvFile("android-release-signing.env")

fun signingSecret(name: String): Provider<String> {
    val fromFile = localSigningEnv[name].orEmpty()
    return providers
        .gradleProperty(name)
        .orElse(providers.environmentVariable(name))
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .orElse(fromFile)
        .filter { it.isNotEmpty() }
}

val releaseKeystorePath = signingSecret("APP_RELEASE_KEYSTORE").orNull
val releaseKeystorePassword = signingSecret("APP_RELEASE_KEYSTORE_PASSWORD").orNull
val releaseKeyAlias = signingSecret("APP_RELEASE_KEY_ALIAS").orNull ?: "pxlcam"
val releaseKeyPassword = signingSecret("APP_RELEASE_KEY_PASSWORD").orNull ?: releaseKeystorePassword
val releaseSigningConfigured =
    releaseKeystorePath != null &&
        releaseKeystorePassword != null &&
        releaseKeyPassword != null

val pxlCamVersionCode = providers.gradleProperty("pxlcam.versionCode").get().toInt()
val pxlCamVersionName = providers.gradleProperty("pxlcam.versionName").get()

android {
    namespace = "com.pxlcam.app"
    compileSdk = Integer.parseInt(findProperty("android.compileSdkVersion") as String)

    defaultConfig {
        applicationId = "com.pxlcam.app"
        minSdk = Integer.parseInt(findProperty("android.minSdkVersion") as String)
        targetSdk = Integer.parseInt(findProperty("android.targetSdkVersion") as String)
        versionCode = pxlCamVersionCode
        versionName = pxlCamVersionName

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        buildConfigField("String", "PRIVACY_POLICY_URL", "\"https://pxlcam.app/privacy\"")
        buildConfigField("int", "DEFAULT_STREAM_PORT", "9876")
    }

    signingConfigs {
        create("release") {
            if (releaseSigningConfigured) {
                storeFile = file(requireNotNull(releaseKeystorePath))
                storePassword = requireNotNull(releaseKeystorePassword)
                keyAlias = releaseKeyAlias
                keyPassword = requireNotNull(releaseKeyPassword)
            }
        }
    }

    buildTypes {
        debug {
            applicationIdSuffix = ""
            isDebuggable = true
        }
        release {
            if (releaseSigningConfigured) {
                signingConfig = signingConfigs.getByName("release")
            } else {
                // Allow local unsigned/debug-signed release verification when keystore is not set;
                // Play bundle upload requires releaseSigningConfigured via validateReleaseSigningConfiguration.
                signingConfig = signingConfigs.getByName("debug")
            }
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            ndk {
                debugSymbolLevel = "SYMBOL_TABLE"
            }
        }
    }

    packaging {
        jniLibs {
            // Mandatory for Google Play 16 KB page-size alignment on Android 15+ (targetSdk 35+)
            useLegacyPackaging = false
        }
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
    }

    buildFeatures {
        buildConfig = true
        viewBinding = true
    }

    sourceSets {
        getByName("main") {
            java.srcDirs("src/main/kotlin", "src/main/java")
        }
    }

    lint {
        disable += setOf("UseTomlInstead")
        abortOnError = false
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_21)
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.splashscreen)
    testImplementation(libs.junit)
}


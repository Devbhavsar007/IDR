// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// JNI Implementation for Android com.idr.navigation.IdrNative

#include "idr_jni_handle.h"
#include <jni.h>
#include <spdlog/spdlog.h>
#include <string>

namespace {

inline idr::android::IdrHandle* getHandle(JNIEnv* env, jobject thiz) {
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeHandle", "J");
    if (!fid) return nullptr;
    jlong ptr = env->GetLongField(thiz, fid);
    return reinterpret_cast<idr::android::IdrHandle*>(ptr);
}

inline void setHandle(JNIEnv* env, jobject thiz, idr::android::IdrHandle* handle) {
    jclass cls = env->GetObjectClass(thiz);
    jfieldID fid = env->GetFieldID(cls, "nativeHandle", "J");
    if (fid) {
        env->SetLongField(thiz, fid, reinterpret_cast<jlong>(handle));
    }
}

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_idr_navigation_IdrNative_create(JNIEnv* env, jobject thiz) {
    try {
        auto* handle = new idr::android::IdrHandle();
        setHandle(env, thiz, handle);
        return JNI_TRUE;
    } catch (const std::exception& e) {
        spdlog::error("JNI create error: {}", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_destroy(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    if (handle) {
        delete handle;
        setHandle(env, thiz, nullptr);
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_setVehicleProfile(JNIEnv* env, jobject thiz, jstring profile) {
    auto* handle = getHandle(env, thiz);
    if (!handle || !profile) return;

    const char* str = env->GetStringUTFChars(profile, nullptr);
    if (str) {
        handle->setVehicleProfile(std::string(str));
        env->ReleaseStringUTFChars(profile, str);
    }
}

JNIEXPORT jboolean JNICALL
Java_com_idr_navigation_IdrNative_loadModel(JNIEnv* env, jobject thiz, jstring model_path) {
    auto* handle = getHandle(env, thiz);
    if (!handle || !model_path) return JNI_FALSE;

    const char* str = env->GetStringUTFChars(model_path, nullptr);
    bool ok = false;
    if (str) {
        ok = handle->loadModel(std::string(str));
        env->ReleaseStringUTFChars(model_path, str);
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_start(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    if (handle) {
        handle->start();
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_stop(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    if (handle) {
        handle->stop();
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_feedImu(
    JNIEnv* env, jobject thiz,
    jlong timestamp_ns,
    jdoubleArray accel,
    jdoubleArray gyro,
    jdoubleArray gravity) {
    auto* handle = getHandle(env, thiz);
    if (!handle || !accel || !gyro) return;

    jdouble* acc_ptr = env->GetDoubleArrayElements(accel, nullptr);
    jdouble* gyr_ptr = env->GetDoubleArrayElements(gyro, nullptr);
    jdouble* grav_ptr = gravity ? env->GetDoubleArrayElements(gravity, nullptr) : nullptr;

    double acc[3] = {acc_ptr[0], acc_ptr[1], acc_ptr[2]};
    double gyr[3] = {gyr_ptr[0], gyr_ptr[1], gyr_ptr[2]};
    double grav[3];
    if (grav_ptr) {
        grav[0] = grav_ptr[0];
        grav[1] = grav_ptr[1];
        grav[2] = grav_ptr[2];
    }

    handle->feedImu(timestamp_ns, acc, gyr, grav_ptr ? grav : nullptr);

    env->ReleaseDoubleArrayElements(accel, acc_ptr, JNI_ABORT);
    env->ReleaseDoubleArrayElements(gyro, gyr_ptr, JNI_ABORT);
    if (grav_ptr) {
        env->ReleaseDoubleArrayElements(gravity, grav_ptr, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_feedGnss(
    JNIEnv* env, jobject thiz,
    jlong timestamp_ns,
    jdouble lat_deg, jdouble lon_deg, jdouble alt_m,
    jfloat speed_mps, jfloat bearing_deg,
    jfloat h_accuracy_m, jfloat v_accuracy_m,
    jint fix_type, jint sat_count) {
    auto* handle = getHandle(env, thiz);
    if (handle) {
        handle->feedGnss(timestamp_ns, lat_deg, lon_deg, alt_m,
                         speed_mps, bearing_deg, h_accuracy_m, v_accuracy_m,
                         fix_type, sat_count);
    }
}

JNIEXPORT void JNICALL
Java_com_idr_navigation_IdrNative_feedBaro(
    JNIEnv* env, jobject thiz,
    jlong timestamp_ns, jdouble pressure_hpa) {
    auto* handle = getHandle(env, thiz);
    if (handle) {
        handle->feedBaro(timestamp_ns, pressure_hpa);
    }
}

JNIEXPORT jobject JNICALL
Java_com_idr_navigation_IdrNative_getNavigationState(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    idr::NavigationState state;
    if (handle) {
        state = handle->getNavigationState();
    }

    jclass cls = env->FindClass("com/idr/navigation/IdrNative$NavigationState");
    if (!cls) return nullptr;

    jmethodID ctor = env->GetMethodID(cls, "<init>", "(DDDDDDDDDDDIDI ZIJDIJ)V");
    if (!ctor) {
        // Try fallback if constructor has different descriptor without spaces
        ctor = env->GetMethodID(cls, "<init>", "(DDDDDDDDDDDDIZIJDIJ)V");
    }
    if (!ctor) return nullptr;

    return env->NewObject(cls, ctor,
        static_cast<jdouble>(state.latitude_deg),
        static_cast<jdouble>(state.longitude_deg),
        static_cast<jdouble>(state.altitude_m),
        static_cast<jdouble>(state.speed_mps),
        static_cast<jdouble>(state.heading_deg),
        static_cast<jdouble>(state.pitch_deg),
        static_cast<jdouble>(state.roll_deg),
        static_cast<jdouble>(state.horizontal_accuracy_m),
        static_cast<jdouble>(state.vertical_accuracy_m),
        static_cast<jdouble>(state.velocity_accuracy_mps),
        static_cast<jdouble>(state.heading_accuracy_deg),
        static_cast<jint>(state.mode),
        static_cast<jdouble>(state.gnss_confidence),
        static_cast<jboolean>(state.is_stationary ? JNI_TRUE : JNI_FALSE),
        static_cast<jint>(state.alignment_quality),
        static_cast<jlong>(state.matched_road_id),
        static_cast<jdouble>(state.matched_road_confidence),
        static_cast<jint>(state.road_level),
        static_cast<jlong>(state.timestamp_ns)
    );
}

JNIEXPORT jobject JNICALL
Java_com_idr_navigation_IdrNative_getDiagnostics(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    idr::android::IdrHandle::Diagnostics d;
    if (handle) {
        d = handle->getDiagnostics();
    }

    jclass cls = env->FindClass("com/idr/navigation/IdrNative$Diagnostics");
    if (!cls) return nullptr;

    jmethodID ctor = env->GetMethodID(cls, "<init>", "(IIIZZDJ)V");
    if (!ctor) return nullptr;

    return env->NewObject(cls, ctor,
        static_cast<jint>(d.mode),
        static_cast<jint>(d.gnss_integrity),
        static_cast<jint>(d.alignment),
        static_cast<jboolean>(d.is_stationary ? JNI_TRUE : JNI_FALSE),
        static_cast<jboolean>(d.model_loaded ? JNI_TRUE : JNI_FALSE),
        static_cast<jdouble>(d.horizontal_accuracy_m),
        static_cast<jlong>(d.imu_count)
    );
}

JNIEXPORT jboolean JNICALL
Java_com_idr_navigation_IdrNative_isRunning(JNIEnv* env, jobject thiz) {
    auto* handle = getHandle(env, thiz);
    if (!handle) return JNI_FALSE;
    return handle->isRunning() ? JNI_TRUE : JNI_FALSE;
}

}  // extern "C"

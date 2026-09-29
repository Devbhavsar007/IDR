// IDR — Unit tests for NeuralInference (buffer and interface).

#include "idr/neural/neural_inference.h"
#include <gtest/gtest.h>

namespace idr {
namespace {

TEST(SensorWindowTest, InitiallyNotFull) {
    SensorWindowBuffer buf(100, 9);
    EXPECT_FALSE(buf.isFull());
}

TEST(SensorWindowTest, FullAfterEnoughSamples) {
    SensorWindowBuffer buf(10, 9);
    for (int i = 0; i < 10; i++) {
        buf.addSample(Vec3d(1, 2, 3), Vec3d(0.1, 0.2, 0.3), Vec3d(0, 0, 9.8));
    }
    EXPECT_TRUE(buf.isFull());
}

TEST(SensorWindowTest, ShapeIsCorrect) {
    SensorWindowBuffer buf(200, 9);
    auto shape = buf.shape();
    EXPECT_EQ(shape[0], 1);     // batch
    EXPECT_EQ(shape[1], 200);   // window
    EXPECT_EQ(shape[2], 9);     // channels
}

TEST(SensorWindowTest, DataIsNotNull) {
    SensorWindowBuffer buf(10, 9);
    buf.addSample(Vec3d::Zero(), Vec3d::Zero(), Vec3d(0, 0, 9.8));
    EXPECT_NE(buf.data(), nullptr);
    EXPECT_EQ(buf.size(), 90u);  // 10 * 9
}

TEST(SensorWindowTest, ResetClearsFull) {
    SensorWindowBuffer buf(5, 9);
    for (int i = 0; i < 5; i++) {
        buf.addSample(Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero());
    }
    EXPECT_TRUE(buf.isFull());
    buf.reset();
    EXPECT_FALSE(buf.isFull());
}

TEST(NeuralInferenceTest, GracefulWithoutModel) {
    NeuralInference inference;
    EXPECT_FALSE(inference.isModelLoaded());

    inference.feedSample(Vec3d::Zero(), Vec3d::Zero(), Vec3d(0, 0, 9.8));
    EXPECT_FALSE(inference.shouldRunInference(100.0));

    auto result = inference.runInference();
    EXPECT_FALSE(result.valid);
}

TEST(NeuralInferenceTest, LogVarToSigma) {
    Vec3d log_var(0.0, 0.0, 0.0);  // log(1) = 0 → σ = exp(0) = 1
    Vec3d sigma = NeuralInference::logVarToSigma(log_var);
    EXPECT_NEAR(sigma.x(), 1.0, 1e-6);
    EXPECT_NEAR(sigma.y(), 1.0, 1e-6);
    EXPECT_NEAR(sigma.z(), 1.0, 1e-6);
}

}  // namespace
}  // namespace idr

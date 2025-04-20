#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <functional>
#include <iomanip>
#include "dual_number.h"

const int NUM_ITERATIONS = 10000000;
const int VECTOR_SIZE = 100;
const int VECTOR_ITERATIONS = 100000;

template<typename DualFunc, typename FloatFunc>
void profileOperation(const std::string& name, DualFunc dualFunc, FloatFunc floatFunc, 
                     dual_number dualArg, float floatArg) {
    std::cout << "Profiling " << name << ":" << std::endl;
    
    double dualTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        dual_number result;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            result = dualFunc(dualArg);
            if (result.value() < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    double floatTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        float result;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            result = floatFunc(floatArg);
            if (result < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    std::cout << "  Float time: " << floatTime << " seconds" << std::endl;
    std::cout << "  Dual time: " << dualTime << " seconds" << std::endl;
    std::cout << "  Overhead factor: " << dualTime / floatTime << "x" << std::endl;
    std::cout << std::endl;
}

template<typename DualFunc, typename FloatFunc>
void profileBinaryOperation(const std::string& name, DualFunc dualFunc, FloatFunc floatFunc, 
                           dual_number dualArg1, dual_number dualArg2,
                           float floatArg1, float floatArg2) {
    std::cout << "Profiling " << name << ":" << std::endl;
    
    double dualTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        dual_number result;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            result = dualFunc(dualArg1, dualArg2);
            if (result.value() < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    double floatTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        float result;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            result = floatFunc(floatArg1, floatArg2);
            if (result < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    std::cout << "  Float time: " << floatTime << " seconds" << std::endl;
    std::cout << "  Dual time: " << dualTime << " seconds" << std::endl;
    std::cout << "  Overhead factor: " << dualTime / floatTime << "x" << std::endl;
    std::cout << std::endl;
}

template<typename VectorFunc, typename ScalarFunc>
void profileVectorOperation(const std::string& name, VectorFunc vectorFunc, ScalarFunc scalarFunc) {
    std::cout << "Profiling vector " << name << " (size = " << VECTOR_SIZE << "):" << std::endl;
    
    std::vector<float> values(VECTOR_SIZE, 1.0f);
    std::vector<float> duals(VECTOR_SIZE, 0.5f);
    
    double dualTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        dual_vector a(values, duals);
        dual_vector result(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
            result = vectorFunc(a);
            if (result[0].value() < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    double floatTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> a(values);
        std::vector<float> result(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
            for (int j = 0; j < VECTOR_SIZE; ++j) {
                result[j] = scalarFunc(a[j]);
            }
            if (result[0] < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    std::cout << "  Float time: " << floatTime << " seconds" << std::endl;
    std::cout << "  Dual time: " << dualTime << " seconds" << std::endl;
    std::cout << "  Overhead factor: " << dualTime / floatTime << "x" << std::endl;
    std::cout << std::endl;
}

template<typename VectorFunc, typename ScalarFunc>
void profileBinaryVectorOperation(const std::string& name, VectorFunc vectorFunc, ScalarFunc scalarFunc) {
    std::cout << "Profiling vector " << name << " (size = " << VECTOR_SIZE << "):" << std::endl;
    
    std::vector<float> values1(VECTOR_SIZE, 1.0f);
    std::vector<float> values2(VECTOR_SIZE, 2.0f);
    std::vector<float> duals1(VECTOR_SIZE, 0.5f);
    std::vector<float> duals2(VECTOR_SIZE, 0.25f);
    
    double dualTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        dual_vector a(values1, duals1);
        dual_vector b(values2, duals2);
        dual_vector result(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
            result = vectorFunc(a, b);
            if (result[0].value() < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    double floatTime = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> a(values1);
        std::vector<float> b(values2);
        std::vector<float> result(VECTOR_SIZE);
        for (int i = 0; i < VECTOR_ITERATIONS; ++i) {
            for (int j = 0; j < VECTOR_SIZE; ++j) {
                result[j] = scalarFunc(a[j], b[j]);
            }
            if (result[0] < 0) std::cout << ""; // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> profiled_time = end - start;
        return profiled_time.count();
    }();
    
    std::cout << "  Float time: " << floatTime << " seconds" << std::endl;
    std::cout << "  Dual time: " << dualTime << " seconds" << std::endl;
    std::cout << "  Overhead factor: " << dualTime / floatTime << "x" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "===== Dual Number Profiling =====" << std::endl;
    std::cout << "Iterations per operation: " << NUM_ITERATIONS << std::endl;
    std::cout << std::endl;
    
    dual_number x(2.0f, 1.0f);
    dual_number y(3.0f, 0.5f);
    float x_float = 2.0f;
    float y_float = 3.0f;
    
    dual_number neg_x(-0.5f, 1.0f);
    float neg_x_float = -0.5f;
    
    profileOperation("sin", 
        [](dual_number a) { return sin(a); },
        [](float a) { return std::sin(a); },
        x, x_float);
        
    profileOperation("cos", 
        [](dual_number a) { return cos(a); },
        [](float a) { return std::cos(a); },
        x, x_float);
        
    profileOperation("exp", 
        [](dual_number a) { return exp(a); },
        [](float a) { return std::exp(a); },
        x, x_float);
        
    profileOperation("ln", 
        [](dual_number a) { return ln(a); },
        [](float a) { return std::log(a); },
        x, x_float);
        
    profileOperation("relu", 
        [](dual_number a) { return relu(a); },
        [](float a) { return std::max(0.0f, a); },
        x, x_float);
        
    profileOperation("sigmoid", 
        [](dual_number a) { return sigmoid(a); },
        [](float a) { return 1.0f / (1.0f + std::exp(-a)); },
        x, x_float);
        
    profileOperation("tanh", 
        [](dual_number a) { return tanh(a); },
        [](float a) { return std::tanh(a); },
        x, x_float);
    
    profileBinaryOperation("addition", 
        [](dual_number a, dual_number b) { return a + b; },
        [](float a, float b) { return a + b; },
        x, y, x_float, y_float);
        
    profileBinaryOperation("subtraction", 
        [](dual_number a, dual_number b) { return a - b; },
        [](float a, float b) { return a - b; },
        x, y, x_float, y_float);
        
    profileBinaryOperation("multiplication", 
        [](dual_number a, dual_number b) { return a * b; },
        [](float a, float b) { return a * b; },
        x, y, x_float, y_float);
    
    profileVectorOperation("sin", 
        [](dual_vector v) { return sin(v); },
        [](float a) { return std::sin(a); });

    profileVectorOperation("cos", 
        [](dual_vector v) { return cos(v); },
        [](float a) { return std::cos(a); });

    profileVectorOperation("exp", 
        [](dual_vector v) { return exp(v); },
        [](float a) { return std::exp(a); });

    profileVectorOperation("ln", 
        [](dual_vector v) { return ln(v); },
        [](float a) { return std::log(a); });
    
    profileVectorOperation("relu", 
        [](dual_vector v) { return relu(v); },
        [](float a) { return std::max(0.0f, a); });

    profileVectorOperation("tanh", 
        [](dual_vector v) { return tanh(v); },
        [](float a) { return std::tanh(a); });
        
    profileVectorOperation("sigmoid", 
        [](dual_vector v) { return sigmoid(v); },
        [](float a) { return 1.0f / (1.0f + std::exp(-a)); });
        
    profileBinaryVectorOperation("addition", 
        [](dual_vector a, dual_vector b) { return a + b; },
        [](float a, float b) { return a + b; });

    profileBinaryVectorOperation("subtraction", 
        [](dual_vector a, dual_vector b) { return a - b; },
        [](float a, float b) { return a - b; });
        
    profileBinaryVectorOperation("multiplication", 
        [](dual_vector a, dual_vector b) { return a * b; },
        [](float a, float b) { return a * b; });
    
    return 0;
}
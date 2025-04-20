#ifndef DUAL_NUMBER_H
#define DUAL_NUMBER_H

#include <vector>
#include <cmath>
#include <iostream>

class dual_number {
private:
    float val;  
    float dl;  

public:
    dual_number() : val(0.0f), dl(0.0f) {}
    dual_number(float value) : val(value), dl(0.0f) {}
    dual_number(float value, float dual) : val(value), dl(dual) {}
    
    float value() const { return val; }
    float dual() const { return dl; }
    
    dual_number operator+(const dual_number& other) const {
        return dual_number(val + other.val, dl + other.dl);
    }
    
    dual_number operator-(const dual_number& other) const {
        return dual_number(val - other.val, dl - other.dl);
    }
    
    dual_number operator*(const dual_number& other) const {
        return dual_number(val * other.val, val * other.dl + dl * other.val);
    }
};

inline dual_number sin(const dual_number& x) {
    return dual_number(std::sin(x.value()), std::cos(x.value()) * x.dual());
}

inline dual_number cos(const dual_number& x) {
    return dual_number(std::cos(x.value()), -std::sin(x.value()) * x.dual());
}

inline dual_number exp(const dual_number& x) {
    float exp_val = std::exp(x.value());
    return dual_number(exp_val, exp_val * x.dual());
}

inline dual_number ln(const dual_number& x) {
    return dual_number(std::log(x.value()), x.dual() / x.value());
}

inline dual_number relu(const dual_number& x) {
    if (x.value() > 0) {
        return dual_number(x.value(), x.dual());
    } else {
        return dual_number(0.0f, 0.0f);
    }
}

inline dual_number sigmoid(const dual_number& x) {
    float exp_neg_val = std::exp(-x.value());
    float sigmoid_val = 1.0f / (1.0f + exp_neg_val);
    float sigmoid_prime = sigmoid_val * (1.0f - sigmoid_val);
    return dual_number(sigmoid_val, sigmoid_prime * x.dual());
}

inline dual_number tanh(const dual_number& x) {
    float tanh_val = std::tanh(x.value());
    return dual_number(tanh_val, (1.0f - tanh_val * tanh_val) * x.dual());
}

class dual_vector {
private:
    std::vector<dual_number> elements;

public:
    dual_vector() {}
    dual_vector(size_t size) : elements(size) {}
    
    dual_vector(const std::vector<float>& values) {
        elements.reserve(values.size());
        for (const auto& val : values) {
            elements.push_back(dual_number(val));
        }
    }
    
    dual_vector(const std::vector<float>& values, const std::vector<float>& duals) {
        elements.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            elements.push_back(dual_number(values[i], duals[i]));
        }
    }
    
    dual_number& operator[](size_t index) {
        return elements[index];
    }
    
    const dual_number& operator[](size_t index) const {
        return elements[index];
    }
    
    size_t size() const {
        return elements.size();
    }
    
    dual_vector operator+(const dual_vector& other) const {
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = elements[i] + other[i];
        }
        return result;
    }
    
    dual_vector operator-(const dual_vector& other) const {
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = elements[i] - other[i];
        }
        return result;
    }
    
    dual_vector operator*(const dual_vector& other) const {
        dual_vector result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = elements[i] * other[i];
        }
        return result;
    }
};

inline dual_vector sin(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sin(x[i]);
    }
    return result;
}

inline dual_vector cos(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = cos(x[i]);
    }
    return result;
}

inline dual_vector exp(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp(x[i]);
    }
    return result;
}

inline dual_vector ln(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = ln(x[i]);
    }
    return result;
}

inline dual_vector relu(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = relu(x[i]);
    }
    return result;
}

inline dual_vector sigmoid(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]);
    }
    return result;
}

inline dual_vector tanh(const dual_vector& x) {
    dual_vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = tanh(x[i]);
    }
    return result;
}

#endif // DUAL_NUMBER_H
#include "SomeClass.h"

SomeClass::SomeClass(float multiplier_) : multiplier(multiplier_) {}

float SomeClass::multiply(float input) {
    return multiplier * input;
}

std::vector<float> SomeClass::multiply_list(std::vector<float> items) {
    for (float &item : items)
        item = multiply(item);
    return items;
}

std::vector<float> SomeClass::multiply_ndarray(std::vector<float> items) {
    for (float &item : items)
        item = multiply(item);
    return items;
}
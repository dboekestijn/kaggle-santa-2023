#ifndef CPP_SOMECLASS_H
#define CPP_SOMECLASS_H

#include <vector>

class SomeClass {
    float multiplier;

public:
    explicit SomeClass(float multiplier_);

    float multiply(float input);

    std::vector<float> multiply_list(std::vector<float> items);

    std::vector<float> multiply_ndarray(std::vector<float> items);
};


#endif //CPP_SOMECLASS_H

#ifndef CPP_SIMULATOR_H
#define CPP_SIMULATOR_H

#include <array>
#include <future>
#include <random>
#include <vector>

struct VectorHash;

class Simulator {
    unsigned int threadCount;
    unsigned int maxSimDepth;
    std::vector<unsigned short> solutionState;
    std::vector<std::vector<unsigned short>> moves;
    std::vector<unsigned short> sortedMoveIdcs;

public:
    Simulator(unsigned int threadCount_, unsigned int maxSimDepth_,
              const std::vector<unsigned short> &solutionState_,
              const std::vector<std::vector<unsigned short>> &moves_);

    [[nodiscard]] unsigned int getThreadCount() const;

    [[nodiscard]] unsigned int getMaxSimDepth() const;

    [[nodiscard]] std::vector<unsigned short> getSolutionState() const;

    [[nodiscard]] std::vector<unsigned short> getMove(unsigned int i) const;

    [[nodiscard]] std::vector<std::vector<unsigned short>> getMoves() const;

    [[nodiscard]] std::vector<unsigned short> getSimulationResults(const std::vector<unsigned short> &fromState) const;

    [[nodiscard]] std::vector<unsigned short> simulate(const std::vector<unsigned short> &fromState) const;

private:
    [[nodiscard]] std::vector<unsigned short> randperm() const;

    static std::vector<unsigned short> applyMove(const std::vector<unsigned short> &state,
                                                 const std::vector<unsigned short> &move);

};


#endif //CPP_SIMULATOR_H

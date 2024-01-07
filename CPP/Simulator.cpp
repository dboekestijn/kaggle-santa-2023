#include "Simulator.h"

#include <algorithm>
#include <unordered_set>
#include <thread>

Simulator::Simulator(const unsigned int threadCount_, const unsigned int maxSimDepth_,
                     const std::vector<unsigned short> &solutionState_,
                     const std::vector<std::vector<unsigned short>> &moves_)
        : threadCount(threadCount_), maxSimDepth(maxSimDepth_),
        solutionState(solutionState_), moves(moves_) {
    sortedMoveIdcs = std::vector<unsigned short>(moves.size());
    for (unsigned int i = 0; i < sortedMoveIdcs.size(); ++i)
        sortedMoveIdcs[i] = (unsigned short) i;
}

// from https://stackoverflow.com/a/29855973
struct VectorHash {
    size_t operator()(const std::vector<unsigned short> &v) const {
        std::hash<int> hasher;
        size_t seed = 0;
        for (unsigned short i : v)
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        return seed;
    }
};

unsigned int Simulator::getThreadCount() const {
    return threadCount;
}

unsigned int Simulator::getMaxSimDepth() const {
    return maxSimDepth;
}

std::vector<unsigned short> Simulator::getSolutionState() const {
    return solutionState;
}

std::vector<unsigned short> Simulator::getMove(unsigned int i) const {
    return moves[i];
}

std::vector<std::vector<unsigned short>> Simulator::getMoves() const {
    return moves;
}

std::vector<unsigned short> Simulator::getSimulationResults(const std::vector<unsigned short> &fromState) const {
    std::list<std::future<std::vector<unsigned short>>> futures;
    for (unsigned int i = 0; i < threadCount; ++i)
        futures.push_back(std::async([this, &fromState]() {
            return Simulator::simulate(fromState);
        }));

    unsigned int lenShortestMovePath = INT_MAX, lenMovePath;
    std::vector<unsigned short> movePath, shortestMovePath = {};
    for (auto &f : futures) {
        movePath = f.get();
        lenMovePath = movePath.size();
        if (lenMovePath == 0)
            continue;
        else if (lenMovePath < lenShortestMovePath) {
            lenShortestMovePath = lenMovePath;
            shortestMovePath = movePath;
        }
    }

    return shortestMovePath;
}

std::vector<unsigned short> Simulator::simulate(const std::vector<unsigned short> &fromState) const {
    std::unordered_set<std::vector<unsigned short>, VectorHash> visitedStates;
    visitedStates.insert(fromState);

    std::vector<unsigned short> movePath;
    std::vector<unsigned short> shuffledMoveIdcs, move;
    std::vector<unsigned short> currentState = fromState, nextState;
    bool foundNextState;
    for (unsigned int simDepth = 0; simDepth < maxSimDepth; ++simDepth) {
        foundNextState = false;
        shuffledMoveIdcs = randperm();
        for (unsigned short moveIdx : shuffledMoveIdcs) {
            move = moves[moveIdx];
            nextState = applyMove(currentState, move);
            if (visitedStates.find(nextState) == visitedStates.end()) {
                visitedStates.insert(nextState);
                movePath.emplace_back(moveIdx);
                currentState = nextState;
                foundNextState = true;
                break;
            }
        }

        if (!foundNextState)
            return currentState == fromState ? std::vector<unsigned short>() : movePath;
        else if (currentState == solutionState)
            return movePath;
    }

    return {};
}

std::vector<unsigned short> Simulator::randperm() const {
    std::vector<unsigned short> result(sortedMoveIdcs);  // copies sortedMoveIdcs into result
    std::random_device rd;
    std::mt19937 g = std::mt19937(rd());
    std::shuffle(result.begin(), result.end(), g); // shuffle result
    return result;
}

std::vector<unsigned short> Simulator::applyMove(const std::vector<unsigned short> &state, const std::vector<unsigned short> &move) {
    std::vector<unsigned short> permutedState(state.size());
    std::transform(move.begin(), move.end(), permutedState.begin(),
                   [&](size_t index) { return state[index]; });
    return permutedState;
}

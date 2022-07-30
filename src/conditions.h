#pragma once

#include <functional>
#include <unordered_map>

#include "agent.h"

using LiveCondition = std::function<const bool(Agent::SP)>;

const bool LiveStrategy(Agent::SP a);
const bool LiveStrategy_InBounds(Agent::SP a);

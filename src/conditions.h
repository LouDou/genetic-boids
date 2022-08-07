#pragma once

#include <functional>
#include <unordered_map>

#include "agent.h"

using LiveCondition = std::function<const bool(Agent::SP)>;

const Numeric ErrorFunction(Agent::SP a);

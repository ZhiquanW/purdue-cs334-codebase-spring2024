//
// Created by mySab on 2018/9/27.
//

#include "Metal.h"
#include "HitInfo.h"

Metal::Metal(const Vector3 &_a, const double &_f) : albedo(_a), fuzz(_f > 1.0 ? 1.0 : _f) {
}

bool Metal::scatter(const Ray &_in_ray, const HitInfo &_hit_info, Vector3 &_attenuation, Ray &_scattered_ray) const {
    return true;
}

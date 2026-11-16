#include <torch/extension.h>
#include "rasterization_api.h"
#include "adam.h"
#include "lazy_adam.h"
#include "filter3d.h"
#include "densification_api.h"
#include "fused_loss.h"

namespace rasterization_api = faster_gs::rasterization;
namespace adam_api = faster_gs::adam;
namespace filter3d_api = faster_gs::filter3d;
namespace densification_api = faster_gs::densification;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rasterization_api::forward_wrapper);
    m.def("backward", &rasterization_api::backward_wrapper);
    m.def("inference", &rasterization_api::inference_wrapper);
    m.def("pruning_scores", &rasterization_api::pruning_scores_wrapper);
    m.def("adam_step", &adam_api::adam_step_wrapper);
    m.def("lazy_adam_step", &adam_api::lazy_adam_step_wrapper);
    m.def("lazy_adam_flush", &adam_api::lazy_adam_flush_wrapper);
    m.def("lazy_adam_mark", &adam_api::lazy_adam_mark_wrapper);
    m.def("update_3d_filter", &filter3d_api::update_3d_filter_wrapper);
    m.def("relocation_adjustment", &densification_api::relocation_wrapper);
    m.def("add_noise", &densification_api::add_noise_wrapper);
    m.def("l1_dssim_forward", &faster_gs::loss::l1_dssim_forward_wrapper);
    m.def("l1_grad_combine", &faster_gs::loss::l1_grad_combine_wrapper);
}

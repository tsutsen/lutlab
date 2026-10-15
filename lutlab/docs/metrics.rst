lutlab.metrics
==============

How to score a LUT. Accuracy (CIEDE2000 / PSNR / SSIM) answers "how close is
the output to the reference?"; speed (ms -> fps) answers "how fast is it?";
power is a coarse wattage ballpark. ``evaluate`` applies and scores in one
call.

.. autosummary::

   lutlab.metrics.accuracy
   lutlab.metrics.e00
   lutlab.metrics.psnr
   lutlab.metrics.ssim
   lutlab.metrics.lut_space_accuracy
   lutlab.metrics.time_apply
   lutlab.metrics.to_fps
   lutlab.metrics.apply_speed
   lutlab.metrics.gpu_power_watts
   lutlab.metrics.power_sample
   lutlab.metrics.cpu_energy_uj
   lutlab.metrics.evaluate

.. automodule:: lutlab.metrics
   :members:

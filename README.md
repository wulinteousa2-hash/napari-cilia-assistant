# napari-cilia-assistant

`napari-cilia-assistant` is a napari plugin for measuring ciliary beat frequency (CBF) from high-speed AVI microscopy videos.

[Watch the demo video](docs/ui.mp4)

## Install

Install conda-forge first:
https://conda-forge.org/download/

Then open a terminal and run:

```bash
conda create -n cilia-assistant python=3.11 -y
conda activate cilia-assistant
conda install -c conda-forge napari pyqt git -y
git clone https://github.com/wulinteousa2-hash/napari-cilia-assistant.git
cd napari-cilia-assistant
pip install -e .
napari
```

In napari, open `Plugins > Cilia Assistant`.

## What It Does

- Opens one AVI video through the napari widget.
- Reads video metadata such as FPS, frame count, size, and duration.
- Loads AVI data as a grayscale `T, Y, X` image stack.
- Lets the user draw/edit a rectangular ROI over active cilia.
- Measures ROI mean-intensity change over time.
- Estimates CBF using FFT and peak-interval checks.
- Creates a kymograph layer from the selected ROI.
- Exports the last ROI signal and FFT spectrum as CSV files.
- Copies or saves the current measurement graph.

## Basic Workflow

1. Open napari.
2. Open `Plugins > Cilia Assistant`.
3. Click **Open AVI**.
4. Confirm the FPS. Correct it manually if the AVI metadata are wrong.
5. Click **Create / Edit ROI Rectangle**.
6. Move/resize the ROI over visibly beating cilia.
7. Set the expected CBF search range, for example `3-25 Hz`.
8. Click **Measure CBF from Selected ROI**.
9. Review the trace, FFT peak, peak-interval result, and kymograph.
10. Export the CSV files if the result is usable.

## Output

- **FFT CBF:** dominant frequency in the selected search range.
- **Peak-interval CBF:** independent check based on repeated peaks in the ROI signal.
- **Kymograph:** visual audit of periodic motion in the ROI.
- **CSV export:** raw ROI time-intensity signal and FFT power spectrum.

## Good Measurement Practice

- Use videos with known FPS.
- Keep temperature, medium, and timing consistent across samples.
- Place the ROI on active cilia, not static tissue, debris, or whole-frame motion.
- Use multiple ROIs/videos and biological replicates for group comparisons.
- Treat whole-frame motion frequency as exploratory only.

## Limitations

This plugin measures beat frequency, not full ciliary waveform or clinical diagnostic beat pattern. A sample can have a normal CBF but abnormal waveform or poor flow generation. Always review the raw video and kymograph before interpreting the number.

## References

1. Chilvers MA, O'Callaghan C. Analysis of ciliary beat pattern and beat frequency using digital high speed imaging: comparison with the photomultiplier and photodiode methods. *Thorax*. 2000;55:314-317. doi:10.1136/thorax.55.4.314
2. Jackson CL, Bottier M. Methods for the assessment of human airway ciliary function. *European Respiratory Journal*. 2022;60:2102300. doi:10.1183/13993003.02300-2021
3. Francis R. A Simple Method for Imaging and Quantifying Respiratory Cilia Motility in Mouse Models. *Methods and Protocols*. 2025;8:113. doi:10.3390/mps8050113

(.venv) PS C:\Users\Moni\Documents\claudeprojects\Microscope> & c:/Users/Moni/Documents/claudeprojects/Microscope/.venv/Scripts/python.exe c:/Users/Moni/Documents/claudeprojects/Microscope/scripts/gpu_smoke_test.py
2025-09-02 04:33:41,414 - INFO - ============================================================
2025-09-02 04:33:41,414 - INFO - M3 GPU Smoke Test
2025-09-02 04:33:41,414 - INFO - ============================================================
2025-09-02 04:33:41,415 - WARNING - Could not run nvidia-smi: [WinError 2] The system cannot find the file specified
2025-09-02 04:33:41,415 - INFO - Found 3 configurations
2025-09-02 04:33:41,415 - INFO - ------------------------------------------------------------
2025-09-02 04:33:41,415 - INFO - Running as_airy...
2025-09-02 04:33:41,415 - INFO - Command: C:\Users\Moni\Documents\claudeprojects\Microscope\.venv\Scripts\python.exe -m optics_sim.cli.m3_run --config c:\Users\Moni\Documents\claudeprojects\Microscope\examples\m3\as_airy.yaml --out c:\Users\Moni\Documents\claudeprojects\Microscope\gpu_smoke_output\as_airy --device cuda --verbose
2025-09-02 04:33:41,444 - ERROR - ✗ as_airy FAILED
2025-09-02 04:33:41,444 - WARNING -   Missing artifacts: ['output.tiff', 'metrics.json', 'perf.json', 'env.json']
2025-09-02 04:33:41,444 - INFO - ------------------------------------------------------------
2025-09-02 04:33:41,444 - INFO - Running bpm_ss_grating...
2025-09-02 04:33:41,444 - INFO - Command: C:\Users\Moni\Documents\claudeprojects\Microscope\.venv\Scripts\python.exe -m optics_sim.cli.m3_run --config c:\Users\Moni\Documents\claudeprojects\Microscope\examples\m3\bpm_ss_grating.yaml --out c:\Users\Moni\Documents\claudeprojects\Microscope\gpu_smoke_output\bpm_ss_grating --device cuda --verbose        
2025-09-02 04:33:41,469 - ERROR - ✗ bpm_ss_grating FAILED
2025-09-02 04:33:41,470 - WARNING -   Missing artifacts: ['output.tiff', 'metrics.json', 'perf.json', 'env.json']
2025-09-02 04:33:41,470 - INFO - ------------------------------------------------------------
2025-09-02 04:33:41,470 - INFO - Running bpm_wa_highNA...
2025-09-02 04:33:41,470 - INFO - Command: C:\Users\Moni\Documents\claudeprojects\Microscope\.venv\Scripts\python.exe -m optics_sim.cli.m3_run --config c:\Users\Moni\Documents\claudeprojects\Microscope\examples\m3\bpm_wa_highNA.yaml --out c:\Users\Moni\Documents\claudeprojects\Microscope\gpu_smoke_output\bpm_wa_highNA --device cuda --verbose
2025-09-02 04:33:41,497 - ERROR - ✗ bpm_wa_highNA FAILED
2025-09-02 04:33:41,498 - WARNING -   Missing artifacts: ['output.tiff', 'metrics.json', 'perf.json', 'env.json']
2025-09-02 04:33:41,498 - INFO - ------------------------------------------------------------
2025-09-02 04:33:41,498 - WARNING - Could not run nvidia-smi: [WinError 2] The system cannot find the file specified
2025-09-02 04:33:41,498 - INFO - ============================================================
2025-09-02 04:33:41,499 - INFO - SUMMARY
2025-09-02 04:33:41,499 - INFO - ============================================================
2025-09-02 04:33:41,499 - INFO - Successful runs: 0/3
2025-09-02 04:33:41,499 - WARNING - ✗ Some artifacts missing
2025-09-02 04:33:41,499 - ERROR - ✗ GPU SMOKE TEST FAILED
(.venv) PS C:\Users\Moni\Documents\claudeprojects\Microscope> 
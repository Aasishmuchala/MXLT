-- MXLT test-scene builder launcher (3dsmaxbatch entry point).
-- Kept as plain ASCII with no BOM: Max's embedded python rejects a BOM-prefixed
-- file, and PowerShell's utf8 writer adds one. Written from bash for that reason.
global MXLT_BUILD_PY = (getFilenamePath (getSourceFileName())) + "build_test_scene.py"
format "[launcher] executing %\n" MXLT_BUILD_PY
python.ExecuteFile MXLT_BUILD_PY
format "[launcher] done\n"

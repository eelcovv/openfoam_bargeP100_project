checkMesh -writeFields "(nonOrthoAngle skewness wallDistance)"
foamToVTK -faceSet nonOrthoFaces
foamToVTK -faceSet skewness
foamToVTK -faceSet wallDistance

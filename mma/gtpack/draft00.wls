Needs["GroupTheory`"]
GTWhichRepresentation[] (* O(3) *)

(* C2x C3x C4x C2z C3z C2A IC3z *)
GTGetMatrix[C3z]
GTGetMatrix[C3z] // MatrixForm
RotationMatrix[- 2 Pi / 3, {0,0,1}] (* same as C3z *)
GTGetMatrix[IC3z] (*TODO*)

(* all representation: O(3) SU(2) SU(2)xS, default is O(3) *)
GTChangeRepresentation["SU(2)"]
GTChangeRepresentation["SU(2)xS"]

C2x \[SmallCircle] C2x (* Ee, equivalent at GTgmat[] *)
GTgmat[C2x, C2x] (* Ee *)
GTgmat[C2x, C3z] (* C2A *)
GTgmat[C3z, C3zi] (* Ee *)

GTGetMatrix[C3z] (* depend on the current representation*)
GTGetRotationMatrix[C3z] (* O(3) representation *)
GTGetSU2Matrix[C3z] (* SU(2) representation *)
GTGetSymbol[RotationMatrix[- 2 Pi / 3, {0,0,1}]] (* C3z *)

GTGetQuaternion[C3z]
GTGetEulerAngles[C3z]

GTInstallAxis
GTInstallAxis["\[Omega]", {1, 2, 3}]

(* tetrahederal point group T *)
group = GTInstallGroup[T] (* Schönflies notation *)
group = GTInstallGroup[\[LeftBracketingBar]23\[RightBracketingBar]] (* Hermann-Mauguin notation *)
z0 = GTCharacterTable[group]
z0[[1]] (* column name *)
z0[[2]] (* table data *)
z0[[3]] (* row name *)

GTCharacterTable[group, GOIrepNotation -> "Mulliken"][[2]]

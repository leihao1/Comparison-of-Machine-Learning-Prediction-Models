1. Title: SGEMM GPU kernel performance

2. Source Information
   -- Creators: Enrique G. Paredes (egparedes@ifi.uzh.ch) and 
      Rafael Ballester-Ripoll (rballester@ifi.uzh.ch)
     -- Visualization and MultiMedia Lab; Department of Informatics;
        University of Zurich; Zurich, 8050; Switzerland   

   -- Donors: Enrique G. Paredes (egparedes@ifi.uzh.ch) and 
      Rafael Ballester-Ripoll (rballester@ifi.uzh.ch)

   -- Date: October, 2017
 
3. Past Usage:
    1. Rafael Ballester-Ripoll, Enrique G. Paredes, Renato Pajarola. 
       "Sobol Tensor Trains for Global Sensitivity Analysis"
       In arXiv Computer Science / Numerical Analysis e-prints, 2017
       (https://128.84.21.199/abs/1712.00233).
       -- Results:
          -- Prediction of the logarithm of the running time 
          -- A fraction of this data set was used to compute a tensor train based
             predictive model and estimate the Sobol sensitivity indices of
             all the parameters

4. Relevant Information:
   -- This data set measures the running time of a matrix-matrix product A*B = C,
      where all matrices have size 2048 x 2048, using a parameterizable 
      SGEMM GPU kernel with 261400 possible parameter combinations. For each
      tested combination, 4 runs were performed and their results are reported
      as the 4 last columns. All times are measured in milliseconds*.

      There are 14 parameter, the first 10 are ordinal and can only take up to 
      4 different powers of two values, and the 4 last variables are binary.
      Out of 1327104 total parameter combinations, only 261400 are feasible 
      (due to various kernel constraints). This data set contains the results
      for all these feasible combinations.

      The experiment was run on a desktop workstation running Ubuntu 16.04 Linux
      with an Intel Core i5 (3.5GHz), 16GB RAM, and a NVidia Geforce GTX 680 4GB 
      GF580 GTX-1.5GB GPU. We use the "gemm_fast" kernel from the automatic 
      OpenCL kernel tuning library "CLTune" (https://github.com/CNugteren/CLTune).

      * Note: for this kind of data sets it is usually better to work with the
      logarithm of the running times (see e.g. Falch and Elster, "Machine learning-based
      auto-tuning for enhanced performance portability of OpenCL applications", 2015).

5. Number of Instances: 241600 

6. Number of Attributes: 18 (14 predictive attributes, 4 goal fields)

7. Attribute Information:
  -- Independent variables:
    1-2. MWG, NWG: per-matrix 2D tiling at workgroup level: {16, 32, 64, 128} (integer)
    3. KWG: inner dimension of 2D tiling at workgroup level: {16, 32} (integer)
    4-5. MDIMC, NDIMC: local workgroup size: {8, 16, 32} (integer)
    6-7. MDIMA, NDIMB: local memory shape: {8, 16, 32} (integer)
    8. KWI: kernel loop unrolling factor: {2, 8} (integer)
    9-10. VWM, VWN: per-matrix vector widths for loading and storing: {1, 2, 4, 8} (integer)
    11-12. STRM, STRN: enable stride for accessing off-chip memory within a 
           single thread: {0, 1} (categorical)
    13-14. SA, SB: per-matrix manual caching of the 2D workgroup tile: {0, 1} (categorical)

  -- Output:
    15-18. Run1, Run2, Run3, Run4: performance times in milliseconds for 4 independent
           runs using the same parameters. They range between 13.25 and 3397.08.    

8. Missing Attribute Values: None

9. Citation requests / acknowledgments:
  If you use this data set, please cite one or both of these refernces:

    -- Rafael Ballester-Ripoll, Enrique G. Paredes, Renato Pajarola.
    Sobol Tensor Trains for Global Sensitivity Analysis.
    In arXiv Computer Science / Numerical Analysis e-prints, 2017
    (https://128.84.21.199/abs/1712.00233).

    -- Cedric Nugteren and Valeriu Codreanu. CLTune: A Generic Auto-Tuner for OpenCL Kernels.
    In: MCSoC: 9th International Symposium on Embedded Multicore/Many-core Systems-on-Chip. IEEE, 2015
    (http://ieeexplore.ieee.org/document/7328205/)

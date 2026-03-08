# Modeling Human Activity States Using Hidden Markov Models

**AUE EDUCATION | ML TECHNIQUES 2 - FORMATIVE 2**

**Team Members:**
- Fadhlullah Abdulazeez (fadhuweb)
- - Ayomide Agbaje (AgbajeCity)
 
  - ---

  ## Project Overview

  This project implements a Hidden Markov Model (HMM) to recognize and classify human activity states from smartphone accelerometer and gyroscope sensor data. The model is trained on real-world motion data collected using the Sensor Logger app and can distinguish between four distinct activities: **Standing**, **Walking**, **Jumping**, and **Still** (no movement).

  ## Key Results

  - **Overall Test Accuracy:** 63.75%
  - - **Average Sensitivity:** 64.35%
    - - **Average Specificity:** 87.94%
      - - **Best Performing Activity:** Jump (90% accuracy, 100% specificity)
        - - **Dataset:** 398 labeled windows across 4 activities
         
          - ---

          ## Repository Structure

          ```
          ml_techniques__2_formative_2/
          ├── data/
          │   ├── Ayomide/              # Motion data collected by Ayomide Agbaje
          │   └── Fadhlullah/           # Motion data collected by Fadhlullah Abdulazeez
          ├── all_windows_features.csv  # Extracted and normalized features (398 samples)
          ├── data_preprocessing.ipynb  # Data preprocessing and feature extraction notebook
          ├── HMM_Activity_Recognition_Complete.ipynb  # Main HMM implementation and evaluation
          └── README.md                 # This file
          ```

          ---

          ## Data Collection & Preprocessing

          ### Collection Methodology
          - **Sampling Rate:** ~50 Hz (harmonized from dual smartphone devices)
          - - **Activities Recorded:** Standing, Walking, Jumping, Still
            - - **Duration per Activity:** 5-10 seconds per recording
              - - **Total Samples:** 398 feature windows (approximately 50 per activity per person)
               
                - ### Windowing Strategy
                - - **Window Size:** 1 second (50 samples at 50 Hz)
                  - - **Overlap:** 50% sliding window
                    - - **Rationale:**
                      -   - Captures 1-3 complete cycles of periodic activities (walking/jumping)
                          -   - Complies with Nyquist frequency requirements for FFT analysis
                              -   - Provides sufficient statistical samples (~100 per window) for feature stability
                                  -   - Creates manageable feature matrix without excessive computational overhead
                                   
                                      - ---

                                      ## Feature Engineering

                                      ### Time-Domain Features (6 features)
                                      1. **Mean Acceleration (mean_a)** - Overall trend and bias in acceleration signals
                                      2. 2. **Variance (var_a)** - Signal variability; discriminates static vs. dynamic motion
                                         3. 3. **Root Mean Square (rms_a)** - Overall signal energy; sensitive to magnitude variations
                                            4. 4. **Signal Magnitude Area (sma_a)** - Total physical effort; high for jumping, low for standing
                                               5. 5. **Mean Gyroscope (mean_g)** - Overall rotational trend
                                                  6. 6. **Variance Gyroscope (var_g)** - Rotational variability across axes
                                                    
                                                     7. ### Frequency-Domain Features (2 features)
                                                     8. 7. **Dominant Frequency (dom_freq)** - Captures activity periodicity (1-2 Hz for walking, higher for jumping)
                                                        8. 8. **Spectral Energy** - Total power in frequency spectrum; distinguishes high-energy activities (jumping) from low-energy (standing/still)
                                                          
                                                           9. ### Normalization
                                                           10. All features were normalized using **Z-score standardization**:
                                                           11. ```
                                                               z = (x - μ) / σ
                                                               ```
                                                               This ensures fair feature contribution, meets Gaussian HMM assumptions, improves convergence, and maintains interpretability.

                                                               ---

                                                               ## Hidden Markov Model Implementation

                                                               ### Model Architecture
                                                               - **Type:** Gaussian HMM (hmmlearn library)
                                                               - - **Hidden States per Activity:** 3 states
                                                                 - - **Emission Distribution:** Gaussian (diagonal covariance)
                                                                   - - **Training Algorithm:** Baum-Welch EM algorithm
                                                                     - - **Convergence Criteria:** Log-likelihood improvement < 1e-4 or max 200 iterations
                                                                      
                                                                       - ### Convergence Results
                                                                       - | Activity | Iterations | Converged |
                                                                       - |----------|-----------|-----------|
                                                                       - | Standing | 15 | ✓ |
                                                                       - | Walking | 21 | ✓ |
                                                                       - | Jumping | 6 | ✓ |
                                                                       - | Still | 4 | ✓ |
                                                                      
                                                                       - ### Viterbi Decoding
                                                                       - For each test sample, the model computes log-likelihood scores under each activity's HMM. The activity with the highest score is assigned as the prediction.
                                                                      
                                                                       - ---

                                                                       ## Performance Evaluation

                                                                       ### Evaluation Metrics

                                                                       | Activity | Sensitivity | Specificity | Accuracy |
                                                                       |----------|------------|------------|----------|
                                                                       | Standing | 45.00% | 83.33% | 73.75% |
                                                                       | Walking | 52.38% | 86.44% | 77.50% |
                                                                       | Jumping | 60.00% | 100.00% | 90.00% |
                                                                       | Still | 100.00% | 81.97% | 86.25% |
                                                                       | **Average** | **64.35%** | **87.94%** | **81.88%** |

                                                                       ### Key Insights
                                                                       1. **Jump Activity Excellence:** Highest performing activity due to distinctive acceleration spikes and high spectral energy
                                                                       2. 2. **Still Activity Perfect Sensitivity:** Near-zero acceleration is a reliable discriminator
                                                                          3. 3. **Stand-Still Confusion:** Primary misclassification due to similar low-variance features
                                                                             4. 4. **Strong Specificity:** 87.94% average indicates low false positive rate
                                                                                5. 5. **Transition Detection:** Walking shows moderate performance due to intermediate feature values in transition zones
                                                                                  
                                                                                   6. ---
                                                                                  
                                                                                   7. ## Visualizations Generated
                                                                                  
                                                                                   8. 1. **Confusion Matrices** - Raw counts and normalized percentages
                                                                                      2. 2. **HMM Transition Matrices** - Per-activity state transition probabilities
                                                                                         3. 3. **Performance Bar Chart** - Sensitivity, specificity, and accuracy comparison
                                                                                            4. 4. **Activity Sequence Plot** - True vs. Viterbi-decoded sequences for 40 test samples
                                                                                              
                                                                                               5. ---

                                                                                               ## Project Files

                                                                                               ### Notebooks
                                                                                               - **`data_preprocessing.ipynb`** - Raw data loading, windowing, feature extraction, and normalization
                                                                                               - - **`HMM_Activity_Recognition_Complete.ipynb`** - HMM training, Viterbi decoding, evaluation, and visualizations
                                                                                                
                                                                                                 - ### Data
                                                                                                 - - **`all_windows_features.csv`** - Final feature matrix (398 × 8) ready for HMM training
                                                                                                   - - **`data/Ayomide/`** - Motion data collected by Ayomide Agbaje
                                                                                                     - - **`data/Fadhlullah/`** - Motion data collected by Fadhlullah Abdulazeez
                                                                                                      
                                                                                                       - ---
                                                                                                       
                                                                                                       ## Limitations & Future Improvements
                                                                                                       
                                                                                                       ### Current Limitations
                                                                                                       - **Limited Dataset:** 398 windows is small by modern ML standards
                                                                                                       - - **Controlled Environment:** Laboratory setting; real-world variations not captured
                                                                                                         - - **Stand-Still Ambiguity:** Both activities produce similar low-energy signals without additional context
                                                                                                           - - **Fixed Architecture:** All activities use 3 states regardless of actual complexity
                                                                                                            
                                                                                                             - ### Recommended Improvements
                                                                                                             - 1. **Short-term:** Increase dataset size, add cross-validation, implement data augmentation
                                                                                                               2. 2. **Medium-term:** Add additional sensors (GPS, barometer), dynamic state selection, hierarchical HMM
                                                                                                                  3. 3. **Long-term:** Deep learning (LSTM/GRU), transfer learning, continuous activity recognition
                                                                                                                    
                                                                                                                     4. ---
                                                                                                                    
                                                                                                                     5. ## How to Run
                                                                                                                    
                                                                                                                     6. 1. **Data Preprocessing:**
                                                                                                                        2.    ```bash
                                                                                                                                 jupyter notebook data_preprocessing.ipynb
                                                                                                                                 ```
                                                                                                                                 
                                                                                                                                 2. **HMM Training & Evaluation:**
                                                                                                                                 3.    ```bash
                                                                                                                                          jupyter notebook HMM_Activity_Recognition_Complete.ipynb
                                                                                                                                          ```
                                                                                                                                       
                                                                                                                                       3. **Requirements:**
                                                                                                                                       4.    - pandas
                                                                                                                                             -    - numpy
                                                                                                                                                  -    - matplotlib
                                                                                                                                                       -    - seaborn
                                                                                                                                                            -    - scikit-learn
                                                                                                                                                                 -    - hmmlearn
                                                                                                                                                                      -    - scipy
                                                                                                                                                                       
                                                                                                                                                                           - ---
                                                                                                                                                                           
                                                                                                                                                                           ## Team Contribution
                                                                                                                                                                           
                                                                                                                                                                           Both team members contributed equally to this project:
                                                                                                                                                                           
                                                                                                                                                                           | Task | Member |
                                                                                                                                                                           |------|--------|
                                                                                                                                                                           | Data Collection (Standing, Walking) | Fadhlullah Abdulazeez |
                                                                                                                                                                           | Data Collection (Jumping, Still) | Ayomide Agbaje |
                                                                                                                                                                           | Data Preprocessing & Feature Extraction | Shared |
                                                                                                                                                                           | HMM Implementation | Shared |
                                                                                                                                                                           | Evaluation & Visualization | Shared |
                                                                                                                                                                           | Documentation | Shared |
                                                                                                                                                                           
                                                                                                                                                                           ---
                                                                                                                                                                           
                                                                                                                                                                           ## References
                                                                                                                                                                           
                                                                                                                                                                           - Baum, L. E., & Petrie, T. (1966). Statistical inference for probabilistic functions of finite state Markov chains. Journal of the Mathematical Analysis and Applications, 12(2), 200-210.
                                                                                                                                                                           - - Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
                                                                                                                                                                             - - Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
                                                                                                                                                                              
                                                                                                                                                                               - ---
                                                                                                                                                                               
                                                                                                                                                                               **Last Updated:** August 3, 2026
                                                                                                                                                                               **Status:** Complete and ready for submission

# Modeling Human Activity States Using Hidden Markov Models

**Course:** AUE EDUCATION | ML TECHNIQUES 2 - FORMATIVE 2

## Team Members

- Fadhlullah Abdulazeez (fadhuweb)
- - Ayomide Agbaje (AgbajeCity)
 
  - ---

  ## Project Overview

  This project implements a Hidden Markov Model (HMM) to recognize and classify human activity states from smartphone accelerometer and gyroscope sensor data. The model is trained on real-world motion data collected using the Sensor Logger app and can distinguish between four distinct activities: **Standing**, **Walking**, **Jumping**, and **Still** (no movement).

  ## Key Results

  - **Overall Test Accuracy:** 63.75%
  - - **Average Sensitivity:** 64.35%
    - - **Average Specificity:** 87.94%
    - **Best Performing Activity:** Jump (90% accuracy, 100% specificity)
    - - **Dataset Size:** 398 labeled windows across 4 activities
     
      - ---

      ## Feature Engineering

      ### Time-Domain Features (6 features)

      1. **Mean Acceleration** – Overall trend and bias in acceleration signals
      2. 2. **Variance (Acceleration)** – Signal variability; discriminates static vs. dynamic motion
         3. 3. **Root Mean Square (RMS)** – Overall signal energy; sensitive to magnitude variations
            4. 4. **Signal Magnitude Area (SMA)** – Total physical effort; high for jumping, low for standing
            5. **Mean Gyroscope** – Overall rotational trend
            6. 6. **Variance (Gyroscope)** – Rotational variability across axes
              
               7. ### Frequency-Domain Features (2 features)

               7. **Dominant Frequency** – Captures activity periodicity (1-2 Hz for walking, higher for jumping)
               8. 8. **Spectral Energy** – Total power in frequency spectrum; distinguishes high-energy activities (jumping) from low-energy (standing/still)

               ### Data Normalization

               All features were normalized using Z-score standardization.

               ---

               ## Repository Structure

               ```
               ├── HMM_Activity_Recognition_Complete.ipynb
               ├── data/
               │   ├── Ayomide/
               │   └── Fadhullah/
               ├── all_windows_features.csv
               └── README.md
               ```

               ---

               ## Performance Evaluation

               ### Evaluation Metrics

               - **Standing:** Sensitivity: 45.00%, Specificity: 83.33%, Accuracy: 73.75%
               - - **Walking:** Sensitivity: 52.38%, Specificity: 86.44%, Accuracy: 77.50%
                 - - **Jumping:** Sensitivity: 60.00%, Specificity: 100.00%, Accuracy: 90.00%
                 - **Still:** Sensitivity: 100.00%, Specificity: 81.97%, Accuracy: 86.25%
                 - - **Average:** Sensitivity: 64.35%, Specificity: 87.94%, Accuracy: 81.88%
                  
                   - ### Key Insights
                  
                   - 1. **Jump Activity Excellence:** Highest performing activity due to distinctive acceleration spikes and high spectral energy
                     2. 2. **Still Activity Perfect Sensitivity:** Near-zero acceleration is a reliable discriminator
                     3. **Stand-Still Confusion:** Primary misclassification due to similar low-variance features
                     4. 4. **Model Robustness:** High specificity (87.94%) indicates strong false positive control
                        5. 
                        ---

                        ## Limitations & Future Improvements

                        ### Current Limitations

                        - **Limited Dataset:** 398 windows is small by modern ML standards
                        - - **Controlled Environment:** Laboratory setting; real-world variations not captured
                          - - **Stand-Still Ambiguity:** Both activities produce similar low-energy signals
                            - - **Fixed Architecture:** All activities use 3 states regardless of actual complexity
                             
                              - ### Proposed Improvements
                             
                              - 1. **Expand Dataset:** Collect more diverse samples with various users and environments
                                2. 2. **Dynamic State Allocation:** Optimize HMM states per activity based on complexity
                                   3. 3. **Additional Sensors:** Incorporate GPS, pressure, or temperature data
                                      4. 4. **Transfer Learning:** Leverage pre-trained models to improve generalization
                                         5. 5. **Real-Time Implementation:** Deploy model for live activity tracking
                                           
                                            6. ---
                                           
                                            7. ## Files & Data
                                           
                                            8. - **Jupyter Notebook:** `HMM_Activity_Recognition_Complete.ipynb`
                                               - - **Features Matrix:** `all_windows_features.csv` – Final feature matrix (398 × 8) ready for HMM training
                                                 - - **Raw Data:**
                                                   -   - `data/Ayomide/` – Motion data collected by Ayomide Agbaje
                                                       -   - `data/Fadhullah/` – Motion data collected by Fadhlullah Abdulazeez
                                                        
                                                           - ---

                                                           ## Requirements

                                                           - pandas
                                                           - - numpy
                                                             - - matplotlib
                                                               - - seaborn
                                                                 - - scikit-learn
                                                                   - - hmmlearn
                                                                    
                                                                     - ---

                                                                     ## How to Run

                                                                     1. Open the Jupyter notebook: `HMM_Activity_Recognition_Complete.ipynb`
                                                                     2. Execute all cells to reproduce results
                                                                     3. 3. Review evaluation metrics in the performance section
                                                                       
                                                                        4. ---
                                                                       
                                                                        5. ## References
                                                                       
                                                                        6. - Baum, L. E., & Petrie, T. (1966). Statistical inference for probabilistic functions of finite state Markov chains. Journal of the Mathematical Analysis and Applications, 12(2), 200-210.
                                                                           - - Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257-286.
                                                                             - - Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.

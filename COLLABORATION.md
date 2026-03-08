Page_Up# Collaboration & Team Contribution

## Team Members
- **Fadhlullah Abdulazeez** (fadhuweb) - Team Lead
- - **Ayomide Agbaje** (AgbajeCity) - Co-Lead
 
  - ## Task Allocation
 
  - ### Data Collection & Preprocessing (Shared)
  - - **Fadhlullah Abdulazeez**: Collected accelerometer and gyroscope data for Standing and Walking activities
    - - **Ayomide Agbaje**: Collected accelerometer and gyroscope data for Jumping and Still activities
      - - **Both**: Harmonized sampling rates (~50 Hz) and preprocessed raw data into feature vectors
       
        - ### Feature Engineering (Shared)
        - - **Both members**: Extracted and normalized 8 features (6 time-domain, 2 frequency-domain)
          -   - Time-domain: mean, variance, RMS, SMA (accelerometer and gyroscope)
              -   - Frequency-domain: dominant frequency, spectral energy
                  - - **Both members**: Applied Z-score normalization with clear statistical justification
                   
                    - ### HMM Implementation (Shared)
                    - - **Both members**: Implemented Gaussian Hidden Markov Model architecture
                      -   - 3 hidden states per activity (Stand, Walk, Jump, Still)
                          -   - Baum-Welch EM algorithm for training
                              -   - Viterbi algorithm for decoding
                               
                                  - ### Evaluation & Visualization (Shared)
                                  - - **Both members**: Evaluated model performance on unseen test data (80 samples)
                                    -   - Computed sensitivity, specificity, and accuracy metrics
                                        -   - Generated confusion matrices and transition probability visualizations
                                            -   - Created performance comparison bar charts
                                                -   - Plotted decoded activity sequences
                                                 
                                                    - ### Documentation & Reporting (Shared)
                                                    - - **Fadhlullah Abdulazeez**: Initial code implementation and notebook development
                                                      - - **Ayomide Agbaje**: Project documentation, README creation, and report finalization
                                                        - - **Both members**: Contributed to report writing, methodology description, and result interpretation
                                                         
                                                          - ## GitHub Contribution History
                                                         
                                                          - | Contributor | Commits | Contribution | Date |
                                                          - |-------------|---------|--------------|------|
                                                          - | fadhuweb | 6 | Data collection, preprocessing, HMM implementation, visualizations | Mar 5, 2026 |
                                                          - | AgbajeCity | 2+ | Documentation, README, analysis documentation | Mar 8, 2026 |
                                                         
                                                          - ## Collaboration Notes
                                                         
                                                          - ✅ **Balanced Effort**: Both team members contributed significantly to all major aspects of the project
                                                          - ✅ **Clear Division**: Data collection was split based on activity type (dynamic vs. static)
                                                          - ✅ **Shared Development**: All technical implementation (feature extraction, HMM, evaluation) was collaborative
                                                          - ✅ **Equal Credit**: Both members invested substantial time in analysis, visualization, and documentation
                                                         
                                                          - ## Key Deliverables
                                                          - - ✅ 398 labeled sensor data windows from 4 activities
                                                            - - ✅ HMM implementation with Baum-Welch training and Viterbi decoding
                                                              - - ✅ Comprehensive evaluation metrics and visualizations
                                                                - - ✅ Professional documentation and project report
                                                                  - - ✅ GitHub repository with clean, organized structure
                                                                   
                                                                    - ---

                                                                   

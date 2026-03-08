# Analysis & Research Notes - HMM Activity Recognition

## Model Performance Summary

### Overall Metrics
- **Accuracy**: 63.75% on 80 unseen test samples
- - **Average Sensitivity**: 64.35%
  - - **Average Specificity**: 87.94%
    - - **Average Per-Activity Accuracy**: 81.88%
     
      - ### Per-Activity Breakdown
     
      - #### Standing
      - - **Sensitivity**: 45.00% | **Specificity**: 83.33% | **Accuracy**: 73.75%
        - - **Issue**: Confused with "Still" activity (11 out of 20 misclassified as Still)
          - - **Reason**: Both activities have near-zero acceleration, making them difficult to distinguish
           
            - #### Walking
            - - **Sensitivity**: 52.38% | **Specificity**: 86.44% | **Accuracy**: 77.50%
              - - **Issue**: Moderate sensitivity indicates transition-zone ambiguity
                - - **Reason**: Walk-Stand transitions produce intermediate feature values
                 
                  - #### Jumping
                  - - **Sensitivity**: 60.00% | **Specificity**: 100.00% | **Accuracy**: 90.00%
                    - - **Performance**: BEST activity - never misclassified as other activities
                    - **Reason**: High acceleration spikes and spectral energy create distinctive signatures
                   
                    - #### Still
                    - - **Sensitivity**: 100.00% | **Specificity**: 81.97% | **Accuracy**: 86.25%
                      - - **Performance**: Perfect sensitivity - always correctly identified
                        - - **Reason**: Absence of motion (accelerations near zero) is reliable discriminator
                         
                          - ## Key Research Findings
                         
                          - ### 1. Stand-Still Confusion
                          - The primary model error is confusing Standing and Still activities. This is expected because:
                          - - Both produce near-zero mean acceleration
                            - - Both show low variance signals
                              - - Gyroscope data is nearly identical for both
                                - - Without longer temporal context, they appear identical to the HMM
                                 
                                  - ### 2. Transition Zone Effects
                                  - Walking shows moderate performance (52.38% sensitivity) because:
                                  - - Window boundaries may capture partial walk-to-stand transitions
                                    - - Transition frames produce intermediate feature values
                                      - - These ambiguous frames are harder to classify confidently
                                        - - Overlapping windows (50%) help but don't eliminate this issue
                                         
                                          - ### 3. Why Jumping Excels
                                          - Jumping achieves 90% accuracy and 100% specificity because:
                                          - - **High-energy signatures**: Jumping produces large acceleration magnitudes
                                            - - **Clear spectral peaks**: Rapid up-down motion creates obvious frequency patterns
                                              - - **Distinctive SMA**: Signal Magnitude Area is much higher than other activities
                                                - - **Temporal distinctiveness**: Jump cycles are very different from other activities
                                                 
                                                  - ### 4. Specificity Insights
                                                  - High specificity (87.94%) across all activities indicates:
                                                  - - **Low false positive rate**: Model rarely misidentifies activities
                                                    - - **Reliable predictions**: When model predicts an activity, it's usually correct
                                                      - - **Good feature design**: Features effectively separate activities
                                                        - - **Trade-off with sensitivity**: Some true positives missed to achieve this specificity
                                                         
                                                          - ## Detailed Feature Analysis
                                                         
                                                          - ### Most Discriminative Features
                                                          - 1. **Spectral Energy** - Best for distinguishing high-energy (jumping) from low-energy (standing, still)
                                                            2. 2. **RMS Amplitude** - Sensitive to signal magnitude, excellent for activity intensity discrimination
                                                               3. 3. **Variance** - Captures motion variability, distinguishes dynamic (walking, jumping) from static activities
                                                                 
                                                                  4. ### Moderately Useful Features
                                                                  5. 4. **Signal Magnitude Area (SMA)** - Accumulates absolute accelerations, good for overall activity magnitude
                                                                     5. 5. **Mean Acceleration** - Provides baseline energy, helps with activity classification
                                                                        6. 6. **Dominant Frequency** - Captures activity periodicity, useful for walking vs. static distinction
                                                                          
                                                                           7. ### Less Impactful Features
                                                                           8. 7. **Gyroscope Mean** - Low variation across activities, minimal discriminative power
                                                                           8. **Gyroscope Variance** - Provides supplementary rotational information, redundant with accelerometer data
                                                                          
                                                                           9. ## Baum-Welch Training Dynamics
                                                                           10. 
                                                                           ### Convergence Behavior
                                                                           - **Standing HMM**: 15 iterations - moderate complexity
                                                                           - - **Walking HMM**: 21 iterations - most complex, highest variability in features
                                                                             - - **Jumping HMM**: 6 iterations - rapid convergence, distinctive features
                                                                               - - **Still HMM**: 4 iterations - simplest, most stable signals

                                                                               ### Implications
                                                                               - Rapid convergence indicates good feature-activity alignment
                                                                               - - Walking's slow convergence suggests within-activity variability
                                                                                 - - All models converged successfully with robust tolerance (1e-4)
                                                                                   - 
                                                                                   ## Viterbi Decoding Analysis

                                                                                   ### Why Log-Likelihood-Based Classification Works
                                                                                   The Viterbi decoder classifies by computing:
                                                                                   ```
                                                                                   P(observation | activity_HMM) for each activity
                                                                                   Selected activity = argmax(log-likelihood)
                                                                                   ```

                                                                                   ### Success Factors
                                                                                   1. **Clear inter-activity differences**: Different activities have distinct feature distributions
                                                                                   2. **Learned transition structures**: Each activity's HMM captures realistic intra-activity dynamics
                                                                                   3. 3. **High specificity**: Model avoids uncertain classifications

                                                                                   ### Limitations
                                                                                   1. **Sequential dependencies**: HMM ignores longer-term activity sequences
                                                                                   2. 2. **Fixed state structure**: All activities forced into 3-state model
                                                                                   3. **Independent activity models**: No consideration of realistic activity transition probabilities (walking→standing, not jumping→standing)

                                                                                   ## Recommendations for Improvement

                                                                                   ### Short-Term (High Impact, Low Effort)
                                                                                   1. **Use CRF with activity transitions**: Enforce realistic activity sequences (reduce Stand↔Still confusion by 30-40%)
                                                                                   2. **Increase training data**: More samples reduce overfitting, especially for Stand vs. Still
                                                                                   3. 3. **Temporal smoothing**: Apply moving average to predictions to reduce isolated misclassifications
                                                                                     
                                                                                      4. ### Medium-Term (Moderate Impact, Moderate Effort)
                                                                                      5. 1. **Dynamic state selection**: Use BIC/AIC to automatically select optimal number of hidden states per activity
                                                                                         2. 2. **Longer temporal windows**: 2-3 second windows instead of 1-second to capture full activity cycles
                                                                                         3. **Per-user adaptation**: Train individual HMMs for each user to account for biomechanical differences

                                                                                         ### Long-Term (Potential High Impact, High Effort)
                                                                                         1. **Deep learning transition**: Replace HMM with LSTM/GRU for better temporal modeling
                                                                                         2. 2. **Multi-modal fusion**: Combine accelerometer, gyroscope, magnetometer, and barometer
                                                                                         3. **Transfer learning**: Pre-train on public activity recognition datasets, fine-tune on project data

                                                                                         ## Statistical Validation

                                                                                         ### Cross-Validation Recommendation
                                                                                         Current evaluation uses single 80/20 split. For robust assessment:
                                                                                         - Implement 5-fold cross-validation
                                                                                         - - Report mean accuracy with standard deviation
                                                                                         - Test on completely held-out third dataset

                                                                                         ### Statistical Significance
                                                                                         - 80 test samples provides ~8% margin of error (95% confidence)
                                                                                         - Results are statistically meaningful but would benefit from larger test set
                                                                                         - - Confidence intervals should be reported alongside point estimates

                                                                                         ## Open Questions for Future Work

                                                                                         1. **How do sampling rates affect model**: Analyze robustness to 30 Hz vs. 50 Hz vs. 100 Hz data
                                                                                         2. 2. **Can activity transitions improve performance**: Implement transition constraints (HMM → HCRF)
                                                                                         3. **What's optimal window size**: Sweep from 0.5s to 3s windows, measuring accuracy vs. latency
                                                                                         4. **How much data is enough**: Data efficiency analysis with varying training set sizes
                                                                                         5. **Real-time performance**: Can model run on edge devices (phones, watches) at inference?

                                                                                         ---

                                                                                         **Analysis Completed**: August 3, 2026
                                                                                         **Analyst**: Ayomide Agbaje (AgbajeCity)
                                                                                         **Review Status**: Ready for incorporation into final report

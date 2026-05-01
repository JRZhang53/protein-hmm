# Figure Captions

## model_selection_bic.png
Train-set BIC for unsupervised HMMs at K=2,3,4,6,8 (lower is better). The BIC penalty grows roughly as O(K^2) through the transition matrix; the curve trades likelihood gain against this penalty.

## model_selection_val_ll.png
Validation log-likelihood per residue versus K. Useful as a held-out cross-check on BIC.

## em_convergence.png
Total training log-likelihood per Baum-Welch iteration for each candidate K. Used to confirm that EM converged (or to flag the cases where it did not).

## emission_heatmap.png / emission_enrichment.png
Per-state emission probabilities and the log2 enrichment of each amino acid relative to the training-set background. Enrichment is the more interpretable view since it removes the dominant frequency baseline.

## transition_heatmap.png
Latent transition matrix for the main unsupervised HMM. Diagonal mass quantifies state persistence (expected dwell length is 1/(1 - p_ii)).

## state_dssp_enrichment.png
P(DSSP class | latent state) on the test set after Hungarian-mapping latent states to H/E/C using the training set. Quantifies how much DSSP signal each unsupervised state captures.

## state_hydrophobicity.png / state_polarity.png
Per-state biochemical summaries: Kyte-Doolittle hydrophobicity (mass-weighted) and the probability mass on polar / charged residues.

## decoded_*.png
Example test proteins with the Viterbi state path overlaid on the DSSP label band, one per family.

## family_transition_distances.png
Pairwise Frobenius distance between family-specific transition matrices after Hungarian alignment of states by emission Jensen-Shannon distance. Without alignment, these distances are dominated by arbitrary state labelling.

## family_stationary_distances.png
Pairwise L1 distance between sorted stationary distributions across family models. Sorting makes the comparison permutation-invariant without requiring alignment.

## cross_family_log_likelihood.png
Per-residue log-likelihood of each family-trained HMM evaluated on every family's test sequences. Diagonal dominance is the indication that families have distinct sequential organisation.

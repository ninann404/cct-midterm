import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load the plant knowledge dataset and prepare it for analysis.
    
    Parameters:
    filepath (str): Path to the CSV file
    
    Returns:
    numpy.ndarray: Binary response matrix (informants x items)
    pandas.DataFrame: Original dataframe for reference
    """
    # Load the data
    df=pd.read_csv(filepath)
    
    # Extract the binary response matrix (excluding the 'Informant' column)
    X = df.iloc[:, 1:].values
    
    return X, df

def run_cct_model(X):
    """
    Implement the Cultural Consensus Theory model using PyMC.
    
    Parameters:
    X (numpy.ndarray): Binary response matrix (informants x items)
    
    Returns:
    arviz.InferenceData: Trace of the MCMC sampling
    """
    N, M = X.shape  # N informants, M items
    
    with pm.Model() as cct_model:
        # Prior for informant competence (D)
        # Using a Beta prior that ensures D is between 0.5 and 1
        # Beta(2,1) places more weight on values above 0.5
        D_raw = pm.Beta("D_raw", alpha=2, beta=1, shape=N)
        # Transform to ensure D is between 0.5 and 1
        D = pm.Deterministic("D", 0.5 + 0.5 * D_raw)
        
        # Prior for consensus answers (Z)
        # Using Bernoulli(0.5) as a non-informative prior
        Z = pm.Bernoulli("Z", p=0.5, shape=M)
        
        # Reshape D for broadcasting
        D_reshaped = D[:, None]  # Shape: (N, 1)
        
        # Calculate probability matrix
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Define likelihood
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)
        
        # Sample from the posterior
        trace = pm.sample(2000, chains=4, tune=1000, return_inferencedata=True)
    
    return trace

def analyze_results(X, trace, df):
    """
    Analyze the results of the CCT model.
    
    Parameters:
    X (numpy.ndarray): Binary response matrix
    trace (arviz.InferenceData): Trace from MCMC sampling
    df (pandas.DataFrame): Original dataframe with informant IDs
    
    Returns:
    tuple: (competence_estimates, consensus_answers, majority_votes)
    """
    # Check convergence
    summary = az.summary(trace)
    print("Convergence diagnostics summary:")
    print(summary['r_hat'].describe())
    
    # Extract posterior samples
    posterior_samples = trace.posterior
    
    # Estimate informant competence
    competence_estimates = posterior_samples['D'].mean(dim=('chain', 'draw')).values
    
    # Get informant IDs
    informant_ids = df['Informant'].values
    
    print("\nInformant competence estimates:")
    for i, (informant, comp) in enumerate(zip(informant_ids, competence_estimates)):
        print(f"{informant}: {comp:.3f}")
    
    # Identify most and least competent informants
    most_competent_idx = np.argmax(competence_estimates)
    least_competent_idx = np.argmin(competence_estimates)
    
    print(f"\nMost competent informant: {informant_ids[most_competent_idx]} (D = {competence_estimates[most_competent_idx]:.3f})")
    print(f"Least competent informant: {informant_ids[least_competent_idx]} (D = {competence_estimates[least_competent_idx]:.3f})")
    
    # Estimate consensus answers
    z_posterior_mean = posterior_samples['Z'].mean(dim=('chain', 'draw')).values
    consensus_answers = np.round(z_posterior_mean).astype(int)
    
    # Get question IDs
    question_ids = df.columns[1:].values
    
    print("\nConsensus answer probabilities:")
    for j, (question, prob, ans) in enumerate(zip(question_ids, z_posterior_mean, consensus_answers)):
        print(f"{question}: {prob:.3f} (consensus answer: {ans})")
    
    # Calculate majority vote
    majority_votes = np.round(X.mean(axis=0)).astype(int)
    
    print("\nComparison with majority vote:")
    differences = 0
    for j, (question, cons, maj) in enumerate(zip(question_ids, consensus_answers, majority_votes)):
        if cons != maj:
            differences += 1
            print(f"{question}: CCT consensus = {cons}, Majority vote = {maj}")
    
    if differences == 0:
        print("No differences between CCT consensus and majority vote.")
    else:
        print(f"\nTotal differences: {differences}/{len(question_ids)}")
    
    return competence_estimates, consensus_answers, majority_votes, informant_ids, question_ids

def visualize_results(trace, competence_estimates, consensus_answers, majority_votes, informant_ids, question_ids):
    """
    Visualize the results of the CCT model.
    
    Parameters:
    trace (arviz.InferenceData): Trace from MCMC sampling
    competence_estimates (numpy.ndarray): Estimated competence for each informant
    consensus_answers (numpy.ndarray): Estimated consensus answers
    majority_votes (numpy.ndarray): Majority vote answers
    informant_ids (numpy.ndarray): IDs of the informants
    question_ids (numpy.ndarray): IDs of the questions
    """
    # Plot posterior distributions for competence
    plt.figure(figsize=(12, 8))
    az.plot_posterior(trace, var_names=['D'])
    plt.title('Posterior Distributions of Informant Competence')
    plt.tight_layout()
    plt.savefig('competence_posterior.png')
    
    # Plot posterior distributions for consensus answers
    plt.figure(figsize=(15, 10))
    az.plot_posterior(trace, var_names=['Z'])
    plt.title('Posterior Distributions of Consensus Answers')
    plt.tight_layout()
    plt.savefig('consensus_posterior.png')
    
    # Plot competence estimates with informant IDs
    plt.figure(figsize=(10, 6))
    sorted_indices = np.argsort(competence_estimates)[::-1]  # Sort in descending order
    sorted_competence = competence_estimates[sorted_indices]
    sorted_informants = informant_ids[sorted_indices]
    
    plt.bar(range(len(sorted_competence)), sorted_competence)
    plt.xticks(range(len(sorted_competence)), sorted_informants, rotation=45)
    plt.xlabel('Informant')
    plt.ylabel('Estimated Competence (D)')
    plt.title('Informant Competence Estimates')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guessing')
    plt.legend()
    plt.tight_layout()
    plt.savefig('competence_estimates.png')
    
    # Compare consensus answers with majority votes
    plt.figure(figsize=(12, 6))
    x = np.arange(len(question_ids))
    width = 0.35
    
    plt.bar(x - width/2, consensus_answers, width, label='CCT Consensus')
    plt.bar(x + width/2, majority_votes, width, label='Majority Vote')
    
    plt.xlabel('Question')
    plt.ylabel('Answer (0 or 1)')
    plt.title('Comparison of CCT Consensus vs. Majority Vote')
    plt.xticks(x, [q.replace('PQ', '') for q in question_ids], rotation=90)
    plt.yticks([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.savefig('consensus_vs_majority.png')
    
    # Plot posterior probabilities for consensus answers
    plt.figure(figsize=(12, 6))
    z_posterior_mean = trace.posterior['Z'].mean(dim=('chain', 'draw')).values
    
    plt.bar(x, z_posterior_mean)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Uncertainty threshold')
    plt.xlabel('Question')
    plt.ylabel('Posterior Probability of Z=1')
    plt.title('Posterior Probabilities for Consensus Answers')
    plt.xticks(x, [q.replace('PQ', '') for q in question_ids], rotation=90)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('consensus_probabilities.png')

def main():
    # Load data
    X, df = load_data('/home/jovyan/Cultural-Consensus-Theory-with-PyMC/cct-midterm/data/plant_knowledge.csv')
    
    # Run CCT model
    trace = run_cct_model(X)
    
    # Analyze results
    competence_estimates, consensus_answers, majority_votes, informant_ids, question_ids = analyze_results(X, trace, df)
    
    # Visualize results
    visualize_results(trace, competence_estimates, consensus_answers, majority_votes, informant_ids, question_ids)
    
    print("\nAnalysis complete. See generated plots for visualizations.")

if __name__ == "__main__":
    main() 
    

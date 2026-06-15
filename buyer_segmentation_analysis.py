"""
Machine Learning-based Buyer Segmentation and Investment Profiling
for Real Estate Market Intelligence

This module handles:
1. Data loading and cleaning
2. Feature engineering and encoding
3. Feature scaling
4. K-Means and Hierarchical Clustering
5. Elbow Method and Silhouette Analysis
6. Cluster interpretation and profiling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BuyerSegmentationPipeline:
    """
    Complete pipeline for buyer segmentation and investment profiling
    """
    
    def __init__(self, clients_path, properties_path):
        """Initialize the pipeline with data paths"""
        self.clients_df = None
        self.properties_df = None
        self.merged_df = None
        self.processed_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.kmeans_model = None
        self.optimal_k = None
        
        # Load data
        self.load_data(clients_path, properties_path)
    
    def load_data(self, clients_path, properties_path):
        """Load clients and properties data"""
        print("Loading data...")
        self.clients_df = pd.read_csv(clients_path)
        self.properties_df = pd.read_csv(properties_path)
        print(f"✓ Clients data loaded: {self.clients_df.shape}")
        print(f"✓ Properties data loaded: {self.properties_df.shape}")
    
    def clean_data(self):
        """Step 1: Data Cleaning"""
        print("\n" + "="*60)
        print("STEP 1: DATA CLEANING")
        print("="*60)
        
        # Clean clients data
        print("\nCleaning clients data...")
        
        # Handle missing values
        print(f"Missing values before cleaning:\n{self.clients_df.isnull().sum()}")
        
        # Drop rows with missing client_id
        self.clients_df = self.clients_df.dropna(subset=['client_id'])
        
        # Fill missing satisfaction_score with median
        self.clients_df['satisfaction_score'].fillna(
            self.clients_df['satisfaction_score'].median(), inplace=True
        )
        
        # Clean date_of_birth format - standardize dates
        self.clients_df['date_of_birth'] = pd.to_datetime(
            self.clients_df['date_of_birth'], errors='coerce'
        )
        
        # Calculate age
        reference_date = pd.Timestamp('2024-01-01')
        self.clients_df['age'] = (reference_date - self.clients_df['date_of_birth']).dt.days / 365.25
        
        # Remove duplicate client entries (keep first)
        initial_count = len(self.clients_df)
        self.clients_df = self.clients_df.drop_duplicates(subset=['client_id'], keep='first')
        print(f"Removed {initial_count - len(self.clients_df)} duplicate clients")
        
        # Clean properties data
        print("\nCleaning properties data...")
        
        # Remove $ and convert sale_price to float
        self.properties_df['sale_price'] = self.properties_df['sale_price'].str.replace('$', '').str.replace(',', '').astype(float)
        
        # Convert transaction_date to datetime
        self.properties_df['transaction_date'] = pd.to_datetime(
            self.properties_df['transaction_date'], format='%d-%m-%Y'
        )
        
        # Merge on client reference
        print("\nMerging clients and properties...")
        self.properties_sold = self.properties_df[
            (self.properties_df['listing_status'] == 'Sold') & 
            (self.properties_df['client_ref'].notna())
        ].copy()
        
        self.merged_df = pd.merge(
            self.clients_df,
            self.properties_sold.groupby('client_ref').agg({
                'sale_price': ['sum', 'mean', 'count'],
                'floor_area_sqft': ['sum', 'mean'],
                'transaction_date': 'max'
            }).reset_index(),
            left_on='client_id',
            right_on='client_ref',
            how='left'
        )
        
        # Flatten multi-level columns
        self.merged_df.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                   for col in self.merged_df.columns.values]
        
        # Fill NaN values for clients without properties
        self.merged_df.fillna(0, inplace=True)
        
        print(f"✓ Data cleaning complete: {self.merged_df.shape}")
        
        return self.merged_df
    
    def feature_engineering(self):
        """Step 2: Feature Engineering and Encoding"""
        print("\n" + "="*60)
        print("STEP 2: FEATURE ENGINEERING & ENCODING")
        print("="*60)
        
        self.processed_df = self.merged_df.copy()
        
        # Create derived features
        print("\nCreating derived features...")
        
        # Investment intensity (investment purpose preference)
        self.processed_df['investment_ratio'] = (
            self.processed_df['acquisition_purpose'].apply(lambda x: 1 if x == 'Investment' else 0)
        )
        
        # Loan dependency
        self.processed_df['loan_ratio'] = (
            self.processed_df['loan_applied'].apply(lambda x: 1 if x == 'Yes' else 0)
        )
        
        # Property portfolio value (from merged data)
        self.processed_df['total_portfolio_value'] = self.processed_df.get('sale_price_sum', 0)
        self.processed_df['avg_property_price'] = self.processed_df.get('sale_price_mean', 0)
        self.processed_df['property_count'] = self.processed_df.get('sale_price_count', 0)
        self.processed_df['total_sqft'] = self.processed_df.get('floor_area_sqft_sum', 0)
        self.processed_df['avg_sqft'] = self.processed_df.get('floor_area_sqft_mean', 0)
        
        # Categorical encoding
        print("Encoding categorical variables...")
        
        # One-Hot Encoding for multi-category features
        categorical_features = ['client_type', 'acquisition_purpose', 'referral_channel']
        self.processed_df = pd.get_dummies(
            self.processed_df, 
            columns=categorical_features, 
            drop_first=False
        )
        
        # Label encoding for binary features
        self.processed_df['gender_encoded'] = LabelEncoder().fit_transform(self.processed_df['gender'])
        self.processed_df['loan_applied_encoded'] = LabelEncoder().fit_transform(self.processed_df['loan_applied'])
        
        print(f"✓ Feature engineering complete: {self.processed_df.shape[1]} features created")
        
        return self.processed_df
    
    def feature_scaling(self):
        """Step 3: Feature Scaling"""
        print("\n" + "="*60)
        print("STEP 3: FEATURE SCALING")
        print("="*60)
        
        # Select numeric features for clustering
        numeric_features = [
            'age', 'satisfaction_score', 'total_portfolio_value', 
            'avg_property_price', 'property_count', 'total_sqft', 'avg_sqft',
            'investment_ratio', 'loan_ratio', 'gender_encoded', 'loan_applied_encoded'
        ]
        
        # Add one-hot encoded features
        onehot_features = [col for col in self.processed_df.columns 
                           if 'client_type_' in col or 'acquisition_purpose_' in col or 'referral_channel_' in col]
        
        clustering_features = numeric_features + onehot_features
        
        # Handle missing values
        X = self.processed_df[clustering_features].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.X_scaled = pd.DataFrame(X_scaled, columns=clustering_features)
        
        print(f"✓ Features scaled: {self.X_scaled.shape}")
        print(f"  Mean: {self.X_scaled.mean().mean():.4f}, Std: {self.X_scaled.std().mean():.4f}")
        
        return self.X_scaled
    
    def elbow_method(self, k_range=range(2, 11)):
        """Elbow Method for optimal K"""
        print("\n" + "="*60)
        print("STEP 4: OPTIMAL CLUSTER SELECTION - ELBOW METHOD")
        print("="*60)
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            inertias.append(kmeans.inertia_)
            
            sil_score = silhouette_score(self.X_scaled, kmeans.labels_)
            silhouette_scores.append(sil_score)
            
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={sil_score:.4f}")
        
        # Plot Elbow Curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Inertia', fontsize=12, fontweight='bold')
        axes[0].set_title('Elbow Method', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        axes[1].set_title('Silhouette Score Analysis', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('elbow_silhouette_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Elbow method plot saved: elbow_silhouette_analysis.png")
        
        # Determine optimal K (maximum silhouette score or elbow point)
        self.optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✓ Optimal K selected: {self.optimal_k}")
        
        return inertias, silhouette_scores
    
    def kmeans_clustering(self, n_clusters=None):
        """K-Means Clustering"""
        print("\n" + "="*60)
        print("STEP 5: K-MEANS CLUSTERING")
        print("="*60)
        
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.processed_df['cluster_kmeans'] = self.kmeans_model.fit_predict(self.X_scaled)
        
        print(f"✓ K-Means clustering complete with {n_clusters} clusters")
        print(f"\nCluster distribution:")
        print(self.processed_df['cluster_kmeans'].value_counts().sort_index())
        
        return self.processed_df['cluster_kmeans']
    
    def hierarchical_clustering(self, n_clusters=None):
        """Hierarchical Clustering"""
        print("\n" + "="*60)
        print("STEP 6: HIERARCHICAL CLUSTERING")
        print("="*60)
        
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(self.X_scaled, method='ward')
        
        # Create dendrogram
        plt.figure(figsize=(14, 8))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)', fontsize=13, fontweight='bold')
        plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
        print("✓ Dendrogram saved: dendrogram.png")
        
        # Extract cluster labels
        from scipy.cluster.hierarchy import fcluster
        self.processed_df['cluster_hierarchical'] = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        print(f"✓ Hierarchical clustering complete with {n_clusters} clusters")
        print(f"\nCluster distribution:")
        print(self.processed_df['cluster_hierarchical'].value_counts().sort_index())
        
        return self.processed_df['cluster_hierarchical']
    
    def silhouette_analysis(self, n_clusters=None):
        """Silhouette Analysis for cluster quality"""
        print("\n" + "="*60)
        print("SILHOUETTE ANALYSIS")
        print("="*60)
        
        if n_clusters is None:
            n_clusters = self.optimal_k
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_lower = 10
        silhouette_vals = silhouette_samples(self.X_scaled, self.processed_df['cluster_kmeans'])
        
        for i in range(n_clusters):
            cluster_silhouette_vals = silhouette_vals[self.processed_df['cluster_kmeans'] == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = plt.cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontweight='bold')
            y_lower = y_upper + 10
        
        avg_silhouette = np.mean(silhouette_vals)
        ax.axvline(x=avg_silhouette, color="red", linestyle="--", label=f'Average: {avg_silhouette:.3f}')
        ax.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cluster Label', fontsize=12, fontweight='bold')
        ax.set_title('Silhouette Plot for K-Means Clustering', fontsize=13, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('silhouette_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Silhouette analysis plot saved: silhouette_analysis.png")
        
        print(f"Average Silhouette Score: {avg_silhouette:.4f}")
        
        return silhouette_vals
    
    def cluster_profiling(self):
        """Interpret and profile each cluster"""
        print("\n" + "="*60)
        print("STEP 7: CLUSTER PROFILING & INTERPRETATION")
        print("="*60)
        
        cluster_profiles = {}
        
        for cluster_id in sorted(self.processed_df['cluster_kmeans'].unique()):
            cluster_data = self.processed_df[self.processed_df['cluster_kmeans'] == cluster_id]
            
            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.processed_df) * 100,
                'avg_age': cluster_data['age'].mean(),
                'avg_satisfaction': cluster_data['satisfaction_score'].mean(),
                'investment_percentage': (cluster_data['acquisition_purpose'] == 'Investment').sum() / len(cluster_data) * 100,
                'loan_percentage': (cluster_data['loan_applied'] == 'Yes').sum() / len(cluster_data) * 100,
                'corporate_percentage': (cluster_data['client_type'] == 'Company').sum() / len(cluster_data) * 100,
                'avg_portfolio_value': cluster_data['total_portfolio_value'].mean(),
                'avg_property_count': cluster_data['property_count'].mean(),
                'avg_sqft': cluster_data['avg_sqft'].mean(),
                'female_percentage': (cluster_data['gender'] == 'F').sum() / len(cluster_data) * 100,
            }
            
            cluster_profiles[f'Cluster_{cluster_id}'] = profile
            
            print(f"\n{'='*50}")
            print(f"CLUSTER {cluster_id}")
            print(f"{'='*50}")
            print(f"Size: {profile['size']} clients ({profile['percentage']:.1f}%)")
            print(f"Avg Age: {profile['avg_age']:.1f} years")
            print(f"Avg Satisfaction Score: {profile['avg_satisfaction']:.2f}/5")
            print(f"Investment Purpose: {profile['investment_percentage']:.1f}%")
            print(f"Loan Applied: {profile['loan_percentage']:.1f}%")
            print(f"Corporate Buyers: {profile['corporate_percentage']:.1f}%")
            print(f"Avg Portfolio Value: ${profile['avg_portfolio_value']:,.2f}")
            print(f"Avg Property Count: {profile['avg_property_count']:.2f}")
            print(f"Avg Property Size: {profile['avg_sqft']:.0f} sqft")
            print(f"Female Buyers: {profile['female_percentage']:.1f}%")
        
        return pd.DataFrame(cluster_profiles).T
    
    def save_results(self, output_file='segmentation_results.csv'):
        """Save segmentation results"""
        output_df = self.processed_df[[
            'client_id', 'client_type', 'gender', 'age', 'country', 'region',
            'acquisition_purpose', 'satisfaction_score', 'loan_applied',
            'total_portfolio_value', 'property_count', 'cluster_kmeans', 'cluster_hierarchical'
        ]].copy()
        
        output_df.to_csv(output_file, index=False)
        print(f"✓ Results saved to {output_file}")
        
        return output_df


# Main execution
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = BuyerSegmentationPipeline(
        clients_path='clients.csv',
        properties_path='properties.csv'
    )
    
    # Execute all steps
    pipeline.clean_data()
    pipeline.feature_engineering()
    X_scaled = pipeline.feature_scaling()
    
    # Optimal cluster selection
    inertias, silhouette_scores = pipeline.elbow_method()
    
    # K-Means clustering
    pipeline.kmeans_clustering()
    
    # Hierarchical clustering
    pipeline.hierarchical_clustering()
    
    # Silhouette analysis
    pipeline.silhouette_analysis()
    
    # Cluster profiling
    profile_df = pipeline.cluster_profiling()
    
    # Save results
    results_df = pipeline.save_results()
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)

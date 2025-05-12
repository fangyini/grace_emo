import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import argparse
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

def load_grace_features(grace_path: str, feature_type: str) -> np.ndarray:
    """Load features from Grace npz files."""
    features = []
    for npz_file in Path(grace_path).glob("*.npz"):
        data = np.load(npz_file)
        try:
            feature = data[feature_type]
        except KeyError:
            print('detected error when reading ', npz_file)
            raise ValueError(f"Unknown feature type: {feature_type}")
        features.append(feature.flatten())
    return np.stack(features)

def load_human_features(human_path: str, feature_type: str) -> np.ndarray:
    """Load features from human npz files."""
    features = []
    for subject_dir in Path(human_path).iterdir():
        if not subject_dir.is_dir():
            continue
        for emotion_dir in subject_dir.iterdir():
            if not emotion_dir.is_dir():
                continue
            level3_dir = emotion_dir / "level_3"
            if not level3_dir.exists():
                continue
            for npz_file in level3_dir.glob("*.npz"):
                data = np.load(npz_file)
                try:
                    feature = data[feature_type]
                except KeyError:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                if feature_type == 'ldmk':
                    feature = feature.reshape(feature.shape[0], -1)
                features.append(feature)
    return np.concatenate(features, axis=0)

def plot_distributions(grace_features: np.ndarray, 
                      human_features: np.ndarray,
                      feature_type: str,
                      output_path: str,
                       working_axis=1): # axis=0: each feature point, each feature vector
    """Create distribution plots comparing Grace and human features."""
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with equal subplot sizes
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    
    # Plot 1: PCA visualization with density
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    grace_pca = pca.fit_transform(grace_features)
    human_pca = pca.fit_transform(human_features)
    
    # Create main PCA plot
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot human data first (background)
    h_human, xe_human, ye_human = np.histogram2d(human_pca[:, 0], human_pca[:, 1], bins=50)
    h_human_smooth = gaussian_filter(h_human, sigma=1.5)
    extent_human = [xe_human[0], xe_human[-1], ye_human[0], ye_human[-1]]
    ax1.imshow(h_human_smooth.T, extent=extent_human, origin='lower',
               cmap='Blues', alpha=0.3, aspect='auto')  # Added aspect='auto' for equal size
    
    # Add scatter plots with adjusted visibility
    ax1.scatter(human_pca[:, 0], human_pca[:, 1], alpha=0.05, color='blue', s=1, label='Human')
    ax1.scatter(grace_pca[:, 0], grace_pca[:, 1], alpha=0.2, color='red', s=3, label='Grace')
    
    ax1.set_title('PCA Visualization', pad=20)
    # Make legend more visible
    ax1.legend(prop={'size': 12}, markerscale=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Plot 2: Feature value distribution (violin plot)
    ax2 = fig.add_subplot(gs[0, 1])
    grace_mean = np.mean(grace_features, axis=working_axis)
    human_mean = np.mean(human_features, axis=working_axis)
    
    # Create violin plot
    violin_data = [grace_mean, human_mean]
    vp = ax2.violinplot(violin_data, positions=[1, 2], showmeans=True)
    # Customize violin plot colors
    vp['bodies'][0].set_color('red')
    vp['bodies'][1].set_color('blue')
    vp['cmeans'].set_color('black')
    
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Grace', 'Human'])
    ax2.set_title('Feature Distribution (Violin Plot)')
    
    # Plot 3: Feature range distribution
    ax3 = fig.add_subplot(gs[1, 0])
    grace_range = np.ptp(grace_features, axis=working_axis)
    human_range = np.ptp(human_features, axis=working_axis)
    
    # Create side-by-side histograms
    bins = np.linspace(min(grace_range.min(), human_range.min()),
                      max(grace_range.max(), human_range.max()), 50)
    ax3.hist(grace_range, weights=np.ones(len(grace_range)) / len(grace_range), bins=bins, alpha=0.5, color='red', label='Grace')
    ax3.hist(human_range, weights=np.ones(len(human_range)) / len(human_range), bins=bins, alpha=0.5, color='blue', label='Human')
    ax3.set_title('Feature Range Distribution')
    ax3.legend(prop={'size': 12})
    
    # Plot 4: Feature mean distribution comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate KDE for both distributions
    grace_kde = gaussian_kde(grace_mean)
    human_kde = gaussian_kde(human_mean)
    
    # Create evaluation points
    x_eval = np.linspace(min(grace_mean.min(), human_mean.min()),
                        max(grace_mean.max(), human_mean.max()), 200)
    
    # Plot KDE
    grace_density = grace_kde(x_eval)
    human_density = human_kde(x_eval)
    
    # Normalize densities
    grace_density = grace_density / grace_density.max()
    human_density = human_density / human_density.max()
    
    ax4.fill_between(x_eval, 0, grace_density, alpha=0.5, color='red', label='Grace')
    ax4.fill_between(x_eval, 0, -human_density, alpha=0.5, color='blue', label='Human')
    ax4.set_title('Feature Mean Distribution Comparison')
    ax4.legend(prop={'size': 12})
    ax4.set_ylim(-1.1, 1.1)
    
    plt.suptitle(f'Feature Distribution Comparison ({feature_type})', fontsize=16, y=0.95)
    # Adjust layout with more space between subplots
    plt.tight_layout()
    # Add extra space at the top for the suptitle
    plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Compare feature distributions between Grace and human data')
    parser.add_argument('--grace_path', type=str, default='/Users/xiaokeai/Documents/HKUST/projects/grace/grace_emo/dataset/updated_gau_1000_features/data/')
    parser.add_argument('--human_path', type=str, default='/Users/xiaokeai/Documents/HKUST/datasets/MEAD/features/visual_features/')
    parser.add_argument('--feature_type', type=str, choices=['ldmk', 'face_embed'], default='face_embed')
    parser.add_argument('--output_path', type=str, default='./feature_distribution_ldmk.png')
    
    args = parser.parse_args()
    
    # Load features
    grace_features = load_grace_features(args.grace_path, args.feature_type)
    human_features = load_human_features(args.human_path, args.feature_type)
    
    print(f"Grace features shape: {grace_features.shape}")
    print(f"Human features shape: {human_features.shape}")
    
    # Plot distributions
    plot_distributions(grace_features, human_features, args.feature_type, args.output_path)
    print(f"Distribution plot saved to {args.output_path}")

if __name__ == "__main__":
    main()
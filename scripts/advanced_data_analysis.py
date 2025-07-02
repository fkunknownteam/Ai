import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataAnalyzer:
    def __init__(self):
        self.data = None
        self.models = {}
        
    def load_sample_data(self):
        """Generate sample dataset for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        age = np.random.normal(35, 10, n_samples)
        income = age * 1000 + np.random.normal(0, 5000, n_samples)
        education_years = np.random.normal(14, 3, n_samples)
        experience = np.maximum(0, age - education_years - 6 + np.random.normal(0, 2, n_samples))
        
        # Create categorical variables
        departments = np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples)
        performance = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.6, 0.2])
        
        self.data = pd.DataFrame({
            'age': age,
            'income': income,
            'education_years': education_years,
            'experience': experience,
            'department': departments,
            'performance': performance
        })
        
        print("✅ Sample dataset loaded successfully!")
        print(f"Dataset shape: {self.data.shape}")
        return self.data.head()
        
    def comprehensive_analysis(self):
        """Perform comprehensive data analysis"""
        if self.data is None:
            self.load_sample_data()
            
        print("📊 COMPREHENSIVE DATA ANALYSIS")
        print("=" * 50)
        
        # Basic statistics
        print("\n📈 DESCRIPTIVE STATISTICS:")
        print(self.data.describe())
        
        # Data types and missing values
        print(f"\n🔍 DATA INFO:")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Data types:\n{self.data.dtypes}")
        
        # Correlation analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print("Strong correlations (>0.5):")
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"  {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_val:.3f}")
        
        # Create visualizations
        self.create_visualizations()
        
        # Perform machine learning analysis
        self.ml_analysis()
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Data Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Age distribution
        axes[0, 0].hist(self.data['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Income vs Age scatter plot
        scatter = axes[0, 1].scatter(self.data['age'], self.data['income'], 
                                   alpha=0.6, c=self.data['experience'], cmap='viridis')
        axes[0, 1].set_title('Income vs Age (colored by Experience)')
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Income')
        plt.colorbar(scatter, ax=axes[0, 1], label='Experience')
        
        # 3. Department distribution
        dept_counts = self.data['department'].value_counts()
        axes[0, 2].pie(dept_counts.values, labels=dept_counts.index, autopct='%1.1f%%')
        axes[0, 2].set_title('Department Distribution')
        
        # 4. Performance by Department
        perf_dept = pd.crosstab(self.data['department'], self.data['performance'])
        perf_dept.plot(kind='bar', ax=axes[1, 0], stacked=True)
        axes[1, 0].set_title('Performance by Department')
        axes[1, 0].set_xlabel('Department')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend(title='Performance')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Correlation heatmap
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        axes[1, 1].set_title('Correlation Heatmap')
        
        # Add correlation values to heatmap
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        # 6. Box plot of income by performance
        performance_order = ['Low', 'Medium', 'High']
        box_data = [self.data[self.data['performance'] == perf]['income'] for perf in performance_order]
        axes[1, 2].boxplot(box_data, labels=performance_order)
        axes[1, 2].set_title('Income Distribution by Performance')
        axes[1, 2].set_xlabel('Performance Level')
        axes[1, 2].set_ylabel('Income')
        
        plt.tight_layout()
        plt.show()
        
        print("📊 Visualizations created successfully!")
        
    def ml_analysis(self):
        """Perform machine learning analysis"""
        print("\n🤖 MACHINE LEARNING ANALYSIS")
        print("=" * 40)
        
        # Prepare data for ML
        # Encode categorical variables
        data_encoded = pd.get_dummies(self.data, columns=['department', 'performance'])
        
        # Regression: Predict income based on other features
        print("\n1️⃣ INCOME PREDICTION (Regression)")
        X_reg = data_encoded.drop(['income'], axis=1)
        y_reg = data_encoded['income']
        
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_reg, y_train_reg)
        lr_pred = lr_model.predict(X_test_reg)
        lr_mse = mean_squared_error(y_test_reg, lr_pred)
        lr_r2 = lr_model.score(X_test_reg, y_test_reg)
        
        print(f"Linear Regression Results:")
        print(f"  Mean Squared Error: {lr_mse:.2f}")
        print(f"  R² Score: {lr_r2:.3f}")
        
        # Feature importance for regression
        feature_importance = pd.DataFrame({
            'feature': X_reg.columns,
            'importance': abs(lr_model.coef_)
        }).sort_values('importance', ascending=False)
        
        print(f"  Top 5 Important Features:")
        for idx, row in feature_importance.head().iterrows():
            print(f"    {row['feature']}: {row['importance']:.2f}")
        
        # Classification: Predict performance based on other features
        print("\n2️⃣ PERFORMANCE PREDICTION (Classification)")
        
        # Prepare classification data
        perf_encoded = {'Low': 0, 'Medium': 1, 'High': 2}
        y_class = self.data['performance'].map(perf_encoded)
        X_class = self.data[['age', 'income', 'education_years', 'experience']]
        
        X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
            X_class, y_class, test_size=0.2, random_state=42
        )
        
        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_class, y_train_class)
        rf_pred = rf_model.predict(X_test_class)
        rf_accuracy = accuracy_score(y_test_class, rf_pred)
        
        print(f"Random Forest Classification Results:")
        print(f"  Accuracy: {rf_accuracy:.3f}")
        
        # Feature importance for classification
        feature_importance_class = pd.DataFrame({
            'feature': X_class.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"  Feature Importance:")
        for idx, row in feature_importance_class.iterrows():
            print(f"    {row['feature']}: {row['importance']:.3f}")
        
        # Store models
        self.models = {
            'income_predictor': lr_model,
            'performance_classifier': rf_model
        }
        
        print("\n✅ Machine Learning analysis completed!")
        
    def generate_insights(self):
        """Generate business insights from the analysis"""
        print("\n💡 KEY INSIGHTS & RECOMMENDATIONS")
        print("=" * 45)
        
        if self.data is None:
            self.load_sample_data()
        
        # Calculate key metrics
        avg_income = self.data['income'].mean()
        income_std = self.data['income'].std()
        high_performers = self.data[self.data['performance'] == 'High']
        
        insights = [
            f"📊 Average income across all employees: ${avg_income:,.0f}",
            f"📈 Income standard deviation: ${income_std:,.0f}",
            f"🏆 High performers represent {len(high_performers)/len(self.data)*100:.1f}% of workforce",
            f"🎓 Average education years: {self.data['education_years'].mean():.1f}",
            f"💼 Most common department: {self.data['department'].mode()[0]}",
        ]
        
        recommendations = [
            "🎯 Focus retention efforts on high-performing employees",
            "📚 Invest in education programs to boost performance",
            "💰 Review compensation structure for income equity",
            "🔄 Implement cross-departmental collaboration programs",
            "📊 Regular performance reviews and feedback systems"
        ]
        
        print("KEY INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")
            
        print("\nRECOMMENDations:")
        for rec in recommendations:
            print(f"  {rec}")
            
        return insights, recommendations

# Demonstration
if __name__ == "__main__":
    analyzer = AdvancedDataAnalyzer()
    
    print("🚀 Advanced Data Analysis System")
    print("=" * 50)
    
    # Load and analyze data
    sample_data = analyzer.load_sample_data()
    print("\nSample data preview:")
    print(sample_data)
    
    # Perform comprehensive analysis
    analyzer.comprehensive_analysis()
    
    # Generate insights
    analyzer.generate_insights()
    
    print("\n✨ Analysis complete! All visualizations and insights have been generated.")

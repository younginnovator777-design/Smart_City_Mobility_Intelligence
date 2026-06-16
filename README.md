# Smart_City_Mobility_Intelligence

An enterprise-grade machine learning framework engineered to analyze and predict urban traffic and passenger demand dynamics using spatial, geometric, and temporal feature architectures. 

This repository implements a highly optimized multi-framework Gradient Boosting ensemble designed for high-accuracy demand regression tasks.

---

## 🏆 Performance Benchmarks & Validation

This system was developed and validated within a large-scale, national smart-city data analytics challenge featuring massive validation subsets and highly competitive baselines.

* **Predictive Accuracy:** Secured an absolute evaluation score of **91.78%** (Coefficient of Determination scaling).
* **Competitive Standing:** Ranked **1,457** out of over **10,000+** engineering teams globally.
* **Scale Metrics:** Evaluated against a total volume of **100,000** competitive solutions.
* **Selection Track:** Successfully cleared the opening selection stage to reach the **Semifinals** of the tracking block.

---

## 💡 Relevance, Applications, & Impact

### Strategic Relevance
Urban transit networks experience volatile bottlenecks that disrupt supply chains, alter passenger booking behaviors, and induce sudden trip cancellations. This project addresses the core challenge by enabling transportation networks to transition from reactive adjustments to proactive mitigation.

### Primary Applications
* **Dynamic Fleet Allocation:** Allows ride-hailing applications and public transit networks to pre-position vehicle fleets in high-density pockets before bottleneck cycles trigger.
* **Last-Mile Supply Chain Optimization:** Empowers e-commerce logistics engines to modify dispatch windows and routing vectors based on predicted micro-regional traffic pressure.
* **Smart Infrastructure Management:** Feeds clean predictive intelligence into municipal traffic control systems to optimize signal timing sequences dynamically.

### Systemic Impact
By capturing non-linear spatial dependencies and cyclic time dynamics, this architecture minimizes vehicle idle times, drops operational overhead costs significantly, and reduces localized carbon emissions through efficient routing.

---

## ⚙️ Feature Engineering Architecture

The pipeline implements a specialized preprocessing paradigm to structuralize raw tabular records into highly predictive geometric and cyclic spaces:

### A. Spatial & Geometric Transforms
* **Geospatial Coordinates:** Decodes raw regional geo-indices to reconstruct precise physical latitude and longitude coordinates.
* **Hierarchical Regional Profiles:** Slices geographic indices to model varying resolutions of spatial density and micro-regional historical congestion trends.
* **Vector Geometry Displacement:** Computes relative continuous spatial distance (`Dist_to_Center`) and polar trajectory angles (`Angle_to_Center`) to a centralized network coordinate point to assist decision tree structures with drawing circular geographic boundaries cleanly.

### B. Temporal & Cyclical Transforms
* **Cyclic Time Encoding:** Deconstructs standard timestamp strings into exact hour and minute markers, mapping them into continuous trigonometric spaces using 24-hour periodic Sine and Cosine functions to maintain midnight-rollover temporal consistency:
$$time_{sin} = \sin\left(\frac{2\pi \times minutes}{1440}\right), \quad time_{cos} = \cos\left(\frac{2\pi \times minutes}{1440}\right)$$
  
* **Operational Peak Windows:** Categorizes operational volatility via binary flags isolating morning rush hour (07:00–09:00), evening rush hour (17:00–19:00), and low-activity nocturnal windows (01:00–04:00).
* **Weekly Macro-Cycles:** Identifies daily progression indices via modulo mapping alongside binary weekend indicators.

### C. Framework-Specific Categorical Processing
Categorical features (`geohash`, `RoadType`, `Weather`, etc.) are processed concurrently into two representation types to feed heterogeneous boosting architectures:
* **Native Category Formats:** Assigned for optimal split search inside LightGBM and XGBoost.
* **String Representations:** Structured explicitly to feed the specialized symmetric categorical handling arrays within CatBoost (`Pool` objects).

---

## 🤖 Model Architecture & Ensemble Strategy

To secure optimal predictive variance reduction, the pipeline utilizes a robust **Stacked Ensemble Framework** trained over a rigorous **5-Fold Cross-Validation Matrix**.

### Framework Implementations
1. **LightGBM Regressor (Weight: 40%):** Outfitted for deep structural categorical mapping utilizing a leaf-wise growth approach over highly parameterized structures (`num_leaves=127`, `max_depth=12`).
2. **XGBoost Regressor (Weight: 30%):** Leverages aggressive, fast-histogram tree split evaluations (`tree_method='hist'`) across continuous feature fields.
3. **CatBoost Regressor (Weight: 30%):** Processes raw string categorical features via symmetric decision tree structures, reducing overfitting during regional trend mapping.

### Optimization & Aggregation Strategy
* **Early Stopping Validation:** Each fold evaluates learning trajectories against an isolated out-of-fold validation block, cutting training up to 150 iterations post-convergence to avoid local overfitting.
* **Power Averaging Consensus ($p=2$):** Instead of simple linear weights, predictions are ensembled via non-linear power averaging:
  $$\text{Final Prediction} = \left( 0.4 \times M_{\text{LGB}}^2 + 0.3 \times M_{\text{XGB}}^2 + 0.3 \times M_{\text{CB}}^2 \right)^{\frac{1}{2}}$$
  This methodology accentuates high-confidence consensus across all three frameworks while penalizing isolated fringe outliers.

---

## 🔒 Repository & Data Privacy Constraints

* **Data Restrictions:** In adherence to strict corporate data protection clauses and non-disclosure terms, the primary underlying data arrays (`train.csv`, `test.csv`) and source code are omitted from this codebase.

---

## 👥 Contributors

* **Atharv Handa**
* **Ansh Patel**
* **Brajesh Upadhyay**

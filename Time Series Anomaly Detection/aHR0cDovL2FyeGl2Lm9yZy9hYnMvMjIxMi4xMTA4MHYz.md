# IS IT WORTH IT? COMPARING SIX DEEP AND CLASSICAL METHODS FOR UNSUPERVISED ANOMALY DETECTION IN TIME SERIES

Ferdinand Rewicki, Joachim Denzler, Julia Niebling

## 🧩 Problem to Solve

시계열 데이터에서 이상(Anomaly)을 탐지하는 것은 시스템 모니터링, 헬스케어, 사이버 보안 등 다양한 분야에서 매우 중요합니다. 그러나 이상에 대한 정의가 상황에 따라 달라지고, 비학술적 문제에서는 레이블링된 훈련 데이터가 부족하여 비지도 학습 방식이 주로 사용됩니다. 이 연구는 다음과 같은 핵심 질문에 답하고자 합니다:

1. **고전적인(Classical) 방법의 해석 가능성을 희생하면서까지 딥러닝 방법의 잠재적으로 우수한 성능을 추구할 가치가 있는가?**
2. **다양한 이상 탐지 방법들이 어떤 유형의 이상을 탐지하는 데 능숙한가?**

기존에는 많은 이상 탐지 방법들이 존재하지만, 특정 이상 유형에 대한 각 방법의 강점과 약점을 파악하고, 적절한 방법을 선택하는 것이 어렵다는 문제가 있습니다.

## ✨ Key Contributions

- UCR Anomaly Archive 벤치마크 데이터셋을 사용하여 시계열 데이터에 대한 6가지 최첨단 이상 탐지 방법(3가지 고전 머신러닝, 3가지 딥러닝)을 포괄적으로 비교했습니다.
- UCR Anomaly Archive에 16가지 고유한 이상 유형을 주석(Annotation)으로 추가하여 벤치마크의 정보를 풍부하게 만들었습니다.
- 딥러닝 방법의 우수한 성능이 고전 방법의 해석 가능성 손실을 정당화하는지, 그리고 각 분석 방법들이 다른 이상 유형을 탐지하는 데 있어 어떤 유사점과 차이점을 보이는지라는 두 가지 중요한 질문에 답했습니다.
- MDI(Maximally Divergent Intervals) 및 MERLIN 방법의 성능에 서브시퀀스(subsequence) 길이의 영향을 분석했습니다.
- RRCF(Robust Random Cut Forest) 방법에서 점(point-wise) 기반 특징과 서브시퀀스 기반 특징을 비교했습니다.

## 📎 Related Works

- **이상 탐지 일반 리뷰:** [15, 17, 18, 19, 20, 21, 22]와 같은 많은 이상 탐지 관련 조사 및 리뷰 논문들이 존재합니다.
- **실험적 비교 연구:**
  - Freeman et al. [7]은 SARIMAX, GLM, Facebook Prophet [23], Matrix Profile [24], Donut [25] 등 12가지 시계열 이상 탐지 방법을 Numenta 벤치마크 [26]를 기반으로 비교했습니다.
  - Graabæk et al. [28]은 협업 로봇 환경에서 k-Nearest-Neighbors, LOF, PCA, One-Class SVM, Autoencoder 등 15가지 이상 탐지 방법을 비교했습니다.
- **고전 및 딥러닝 이상 탐지 방법 리뷰:** Ruff et al. [1]은 밀도 추정 및 확률 모델, 원-클래스 분류, 재구성 모델로 방법을 분류하고, 각 범주별 고전 및 딥러닝 방법을 제시하며 통일된 관점을 제공했습니다.

## 🛠️ Methodology

이 연구에서는 3가지 고전 머신러닝 방법과 3가지 딥러닝 방법을 선택하여 비지도 이상 탐지 성능을 비교했습니다.

### **분석 방법 (6가지)**

- **고전 머신러닝 방법:**
  - **Robust Random Cut Forest (RRCF)** [9]: Isolation Forest [29]의 변형으로, 데이터 스트림에 적용 가능합니다. 데이터를 재귀적으로 분할하여 단일 지점을 격리하며, Collusive Displacement를 이상 점수로 사용합니다.
  - **Maximally Divergent Intervals (MDI)** [10]: 밀도 기반의 오프라인 이상 탐지 방법으로, 서브시퀀스 $S$의 확률 밀도 $p_S$를 나머지 시계열 $\Omega(S)$의 밀도 $p_\Omega$와 Kullback-Leibler (KL) 발산 $D(p_S, p_\Omega)$을 사용하여 비교합니다. Hotelling's $T^2$ 방법을 기반으로 하는 서브시퀀스 제안 기법을 사용합니다.
  - **MERLIN** [11]: 디스코드(Discord) 발견 기반의 오프라인 이상 탐지 방법입니다. 가장 가까운 비자기-일치(non-self match)로부터 가장 큰 거리(z-정규화된 유클리드 거리)를 가진 서브시퀀스를 이상으로 간주합니다. 적절한 파라미터 $r$을 찾기 위한 구조화된 탐색 절차를 제공합니다.
- **딥러닝 방법:**
  - **Autoencoder (AE)** [34]: 인코더와 디코더 네트워크로 구성된 신경망으로, 입력 데이터를 재구성하도록 훈련됩니다. 시계열의 "정상" 프로파일을 학습하고, 높은 재구성 오차 $\Delta(x, g(f(x)))$를 보이는 입력 시퀀스를 이상으로 탐지합니다.
  - **Graph Augmented Normalizing Flows (GANF)** [12]: 정규화 흐름(Normalizing Flows)을 사용하여 시계열의 밀도를 추정하는 방법입니다. 다변량 시계열 간의 인과 관계를 모델링하기 위해 베이시안 네트워크를 통합하며, 낮은 밀도 영역을 이상으로 간주합니다.
  - **Transformer Network for Anomaly Detection (TranAD)** [13]: Transformer 모델 [37] 기반의 이상 탐지 방법으로, 어텐션 기반 변환을 통해 입력을 재구성하는 것을 학습합니다. 2단계 훈련을 거치며, 재구성 손실을 이상 점수로 사용합니다.

### **벤치마크 데이터셋**

- **UCR Anomaly Archive** [14]: 인간 의학, 산업, 생물학, 기상학 등 4가지 도메인의 250개 단변량 시계열로 구성됩니다. 인공적으로 주입된 16가지 고유한 이상 유형(예: Amplitude Change, Flat, Frequency Change, Local Drop, Local Peak, Missing Drop, Missing Peak, Noise, Outlier, Reversed, Sampling Rate, Signal Shift, Smoothed Increase, Steep Increase, Time Shift, Time Warping, Unusual Pattern)으로 주석 처리되었습니다.

### **실험 설정**

- **전처리:** 모든 시계열 데이터는 $[0, 1]$ 범위로 정규화되었습니다. AE, GANF, TranAD는 고정된 길이 $L$의 서브시퀀스를 사용하며, MDI와 MERLIN은 $L_{min}$에서 $L_{max}$ 범위의 서브시퀀스 길이를 사용합니다.
- **하이퍼파라미터 튜닝:** Bayesian 최적화를 사용하여 25개 시계열에서 F1 점수를 최적화 목표로 하이퍼파라미터를 조정했습니다.
- **이상 점수 분류:** Peak Over Threshold (POT) [44] 방법을 사용하여 이상 점수를 정상/이상으로 분류하는 임계값(Threshold)을 결정했습니다. MERLIN은 이진 레이블을 직접 반환합니다.
- **평가 지표:**
  - **AUC ROC (Area Under the Receiver Operating Characteristic curve):** 이진 분류기의 클래스 분리 능력을 측정합니다. (불균형 문제에는 덜 적합하지만, 문헌의 광범위한 사용으로 인해 보고)
  - **F1 Score:** 정밀도(Precision)와 재현율(Recall)의 조화 평균으로, 임계값의 품질을 나타냅니다.
  - **UCR Score:** UCR Anomaly Archive에서 권장하는 이진 지표로, 시계열 내 유일한 이상이 가장 높은 이상 점수를 가졌는지(1) 여부를 나타냅니다.
- **특수 실험:**
  - **MDI 및 MERLIN의 서브시퀀스 길이 영향:** $L_{min}=75, L_{max}=125$ 고정(기준선) 외에, 참 이상 길이의 $\pm 25\%$로 동적으로 설정하거나, $L=100$으로 고정하는 전략을 비교했습니다.
  - **RRCF의 슬라이딩 윈도우 통계:** 원본 시계열(점 기반)에 RRCF를 적용한 것과, 슬라이딩 윈도우의 통계적 특징(최소, 최대, 변동 계수, 처음 네 순간: 평균, 분산, 왜도, 첨도) 벡터에 RRCF를 적용한 것을 비교했습니다.

## 📊 Results

- **전반적인 성능 (종합 비교):**
  - **고전 머신러닝 방법이 딥러닝 방법보다 전반적으로 우수한 성능을 보였습니다.**
  - MDI는 가장 높은 AUC ROC (0.66)와 UCR Score (0.47)를 달성했으며, 평균 런타임도 74초로 가장 빨랐습니다.
  - MERLIN은 가장 높은 F1 Score (0.27)를 기록했지만, 런타임은 291초로 가장 길었습니다.
  - RRCF는 F1 Score 0.07, UCR Score 0.03으로 가장 낮은 성능을 보였습니다.
  - 딥러닝 방법 중에는 GANF가 가장 우수했으며 (AUC ROC 0.63, F1 0.23, UCR 0.43), 런타임은 109초였습니다. AE와 TranAD는 GANF보다 낮은 성능을 보였습니다.
- **이상 유형별 성능:**
  - 모든 방법(RRCF 제외)은 'steep increase' 유형의 이상을 잘 탐지했습니다. 특히 MDI와 MERLIN은 UCR Score 1.0을 기록했습니다.
  - 'smoothed increase' 이상은 RRCF(점 기반)가 유일하게 UCR Score 1.0으로 잘 탐지했습니다. (TranAD도 5/6 사례에서 탐지)
  - 'outlier' 이상은 GANF와 TranAD가 가장 잘 탐지했습니다.
  - 'noise' 이상은 MDI가 모든 사례에서 가장 높은 이상 점수로 탐지하며 가장 좋은 성능을 보였습니다.
  - MDI와 MERLIN은 함께 16가지 이상 유형 중 11가지를 탐지했습니다.
  - 'flat', 'reversed', 'time shift', 'unusual pattern' 유형은 어떤 방법으로도 안정적으로 탐지되지 않았습니다 (UCR Score 0.5 미만).
- **MDI 및 MERLIN의 서브시퀀스 길이 영향:**
  - 참 이상 길이에 기반한 동적 서브시퀀스 길이 설정은 MDI와 MERLIN 모두에서 AUC ROC와 F1 Score를 증가시켰습니다.
  - MDI의 UCR Score는 기준선 설정에서 가장 높았지만, 동적 설정과 고정 100 길이 설정 간의 차이는 작았습니다.
- **RRCF의 슬라이딩 윈도우 통계 적용:**
  - 슬라이딩 윈도우 통계 기반의 특징(RRCF@sequences)을 사용했을 때, RRCF의 AUC ROC는 0.56에서 0.7로, UCR Score는 0.03에서 0.15로 크게 향상되었습니다.
  - RRCF@sequences는 'steep increase' 이상을 탐지할 수 있게 되었지만, 'smoothed increase' 이상은 탐지하지 못했습니다.

## 🧠 Insights & Discussion

- **고전 방법의 우위:** 연구 결과는 고전적인 머신러닝 방법(MDI, MERLIN)이 딥러닝 방법보다 시계열 이상 탐지에서 전반적으로 우수한 성능을 보인다는 중요한 시사점을 제공합니다. 이는 딥러닝의 복잡성과 잠재적 성능 이점에도 불구하고, 고전 방법이 해석 가능성을 유지하면서 더 나은 결과를 낼 수 있음을 보여줍니다.
- **이상 유형별 강점:** 각 방법은 특정 이상 유형에 대해 다른 강점을 가집니다. 예를 들어, RRCF는 단일 지점 격리 원리로 인해 'smoothed increase'와 같은 미묘한 이상을 잘 탐지했지만, 서브시퀀스 기반 특징을 사용하면 'steep increase'와 같은 집단 이상 탐지 능력이 향상됩니다. 이는 이상 탐지 시 도메인 지식과 이상 유형에 따른 방법 선택이 중요함을 강조합니다.
- **하이퍼파라미터 튜닝의 중요성:** 딥러닝 방법은 많은 하이퍼파라미터에 민감하며, 적절한 튜닝 없이는 성능 저하가 발생할 수 있습니다. 반면 MDI와 MERLIN은 서브시퀀스 길이와 같은 몇몇 핵심 파라미터만 설정하면 되므로 비교적 견고한 성능을 보입니다.
- **런타임 및 실시간 적용:** MDI는 C++ 구현 덕분에 가장 빠른 런타임을 보였습니다. MERLIN은 Python 구현과 복잡한 디스코드 발견 알고리즘으로 인해 런타임이 길었습니다. 대부분의 딥러닝 방법은 훈련 후 온라인 적용이 가능하지만, MDI와 MERLIN의 현재 버전은 주로 오프라인 설정에 적합합니다.
- **평가 지표의 해석:** AUC ROC는 불균형 데이터에 덜 적합하며, F1 Score와 UCR Score를 함께 고려해야 합니다. 높은 UCR Score와 낮은 F1 Score는 실제 이상은 탐지했지만 오탐(false positive)이 많거나 탐지된 이상 길이가 실제와 다르기 때문일 수 있습니다.
- **한계점:** 'flat', 'reversed', 'time shift', 'unusual pattern'과 같은 일부 이상 유형은 모든 분석 방법에서 탐지에 어려움을 겪었습니다. 이는 향후 연구에서 더 심층적인 이론적 분석이 필요함을 시사합니다.

## 📌 TL;DR

이 연구는 시계열 데이터에서 이상 탐지를 위해 6가지 비지도 학습 방법(3가지 고전 머신러닝: RRCF, MDI, MERLIN; 3가지 딥러닝: AE, GANF, TranAD)의 성능을 UCR Anomaly Archive 벤치마크 데이터셋에서 비교했습니다. 핵심 질문은 딥러닝의 성능이 고전 방법의 해석 가능성 상실을 정당화하는지, 그리고 각 방법이 어떤 유형의 이상을 잘 탐지하는지였습니다. 실험 결과, **MDI와 MERLIN 같은 고전 머신러닝 방법이 딥러닝 방법보다 전반적으로 우수한 성능을 보였습니다.** 특히 MDI는 AUC ROC와 UCR 점수에서, MERLIN은 F1 점수에서 가장 높았습니다. RRCF는 단일 지점 기반으로는 성능이 낮았지만, 슬라이딩 윈도우 통계 기반의 시퀀스 특징을 사용할 때 성능이 크게 향상되었습니다. GANF는 딥러닝 방법 중 가장 우수했지만, 고전 방법에는 미치지 못했습니다. 연구는 이상 탐지 시 방법의 선택이 이상 유형 및 사용 가능한 도메인 지식(예: 서브시퀀스 길이)에 따라 달라져야 함을 강조하며, 고전 방법이 복잡한 딥러닝 모델에 비해 해석 가능성을 유지하면서도 뛰어난 성능을 제공할 수 있음을 보여주었습니다.

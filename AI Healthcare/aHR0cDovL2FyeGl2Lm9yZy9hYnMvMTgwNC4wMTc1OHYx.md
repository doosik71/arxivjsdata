# Processing of Electronic Health Records using Deep Learning: A review

Venet Osmani, Li Li, Matteo Danieletto, Benjamin Glicksberg, Joel Dudley, Oscar Mayora (2017/2018 추정)

## 🧩 Problem to Solve

본 논문은 전 세계 의료 시스템에 도입된 전자 건강 기록(Electronic Health Records, EHR) 및 전자 의료 기록(Electronic Medical Records, EMR)의 방대한 데이터를 어떻게 효율적으로 처리하고 활용할 것인가에 대한 문제를 다룬다. 

EMR 데이터는 환자의 진단, 처방, 수술 기록, 검사 결과 등 종단적(longitudinal) 데이터를 포함하고 있어 정밀 의료(Precision Medicine)를 실현할 수 있는 핵심 자원이다. 그러나 EMR 데이터는 본질적으로 다차원적이고, 이질적(heterogeneous)이며, 데이터가 비어 있는 희소성(sparseness)과 노이즈가 많은 특성을 가지고 있다. 기존의 전통적인 머신러닝 방식은 전문가의 수동한 특성 추출(feature extraction)과 선택 과정에 크게 의존하므로, 대규모의 복잡한 EMR 데이터를 자동으로 분석하고 확장하는 데 한계가 있다.

따라서 본 연구의 목표는 딥러닝(Deep Learning) 기술이 이러한 EMR 데이터의 자동 처리를 어떻게 가능하게 하며, 특히 만성 질환의 진화 이해와 질병 발생 위험 및 합병증 예측에 어떻게 기여하고 있는지를 종합적으로 검토하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 최신 딥러닝 아키텍처가 EMR 데이터의 복잡한 특성을 어떻게 극복하고 임상적 통찰을 제공하는지를 체계적으로 분석한 점이다. 

가장 중심적인 아이디어는 딥러닝의 '자동 특성 학습(automatic feature learning)' 능력을 통해, 사람이 직접 정의하지 않아도 데이터로부터 최적의 표현(representation)을 직접 학습함으로써 EMR의 노이즈와 희소성 문제를 해결할 수 있다는 것이다. 논문은 이를 위해 질병 모델링(Disease Modelling)과 딥러닝 아키텍처(Deep Learning Architectures)라는 두 가지 관점에서 기존 연구들을 분류하고 분석하여 제시한다.

## 📎 Related Works

논문은 전통적인 머신러닝 접근 방식과 딥러닝 방식의 차별점을 명확히 제시한다.

1. **전통적 머신러닝의 한계**: 원시 데이터(raw form)를 직접 처리하는 능력이 부족하며, 도메인 전문가가 특성 추출 및 선택 과정을 설계해야 한다. 이는 데이터의 규모가 커질수록 확장성(scalability) 문제를 야기한다.
2. **기존의 의료 데이터 분석**: 
    - **MEWS (Modified Early Warning Score)**: 6가지 주요 생체 신호를 통해 환자 상태 악화를 추적하는 전통적 알고리즘이다.
    - **GWAS (Genome-Wide Association Study)**: 유전체 전반의 연관성 분석을 수행하며, 최근에는 EMR과 연계하여 표현형 선택의 자유도를 높인 **PheWAS (Phenome-Wide Association Study)**로 확장되고 있다.

본 논문은 이러한 기존 방식들이 가진 수동적 분석의 한계를 딥러닝의 계층적 구조를 통한 추상적 특성 조합 생성 능력으로 해결할 수 있음을 강조한다.

## 🛠️ Methodology

본 논문은 리뷰 논문으로서 PRISMA(Preferred Reporting Items for Systematic Reviews and Meta-Analysis) 가이드라인을 준수하여 문헌 조사를 수행하였다. Google Scholar, PubMed, IEEE, ACM 데이터베이스에서 1,790개의 논문을 검색하였고, 최종적으로 36개의 핵심 논문을 선정하여 분석하였다.

논문에서 소개된 주요 방법론 및 모델 아키텍처는 다음과 같다.

### 1. 질병 모델링 접근법 (Disease Modelling)
- **TDA (Topological Data Analysis)**: 제2형 당뇨병(T2D)의 하위 그룹을 식별하기 위해 사용되었다. 
    - **절차**: 환자 간의 유사도를 측정하기 위해 코사인 유사도(cosine similarity)를 계산하고, $L$-infinity centrality와 주성분 특이값 분해(Principal Metric SVD)를 적용하여 환자-환자 네트워크를 생성한다.
    - **목적**: 이를 통해 질병의 이질성을 분석하고 생물학적, 임상적 특성이 다른 하위 집단을 구분(stratification)한다.
- **Deep Patient**: Stacked Denoising Autoencoders (SDAs)를 사용하여 환자의 일반적인 딥 표현(deep representation)을 학습한다.
    - **구조**: 3개의 층으로 구성된 SDA가 각 층에서 데이터의 은닉 표현(hidden representation)을 매핑하여 미래의 질병 발생 위험을 예측한다.

### 2. 딥러닝 아키텍처 (DL Architectures)
논문은 EMR 분석에 사용된 다양한 신경망 구조를 설명한다.

- **RNN 및 변형 모델 (LSTM, GRU)**: 데이터의 시계열적 특성을 처리하기 위해 사용된다.
    - **DeepCare**: 진단 벡터와 중재(intervention) 벡터를 결합하여 LSTM 네트워크에 입력함으로써 다음 의료 중재 시점을 예측한다.
    - **Doctor AI**: GRU(Gated Recurrent Unit)를 사용하여 의사의 의사결정 과정을 모델링하며, 미래 질병 및 약물 처방을 예측한다.
- **CNN (Convolutional Neural Networks)**:
    - **구조**: 시간(time)과 임상 이벤트(clinical event)를 두 차원으로 하는 시간 행렬(temporal matrix)을 입력으로 사용한다. 특히 'Slow Fusion' 기법을 적용한 CNN이 심부전 및 만성 폐쇄성 폐질환(COPD) 예측에 효과적임을 언급한다.
- **RBM (Restricted Boltzmann Machines)**: 자살 위험 층화(risk stratification) 분석 등에 사용되었다.

## 📊 Results

본 논문에서 분석한 주요 실험 결과는 다음과 같다.

1. **Deep Patient의 성능**:
    - 78개 질병에 대해 예측 모델을 구축한 결과, 평균 AUC-ROC $0.773$을 기록하여 독립 성분 분석(ICA)의 $0.695$보다 월등한 성능을 보였다.
    - 질병별로 예측력의 차이가 있었는데, 당뇨 합병증 예측은 $\text{AUC-ROC} = 0.907$으로 매우 높았으나, 지질 대사 장애는 $0.561$로 낮았다.
    - 정신 질환 관련 예측은 대부분 $\text{AUC-ROC} > 0.6$이었으며, 주의력 결핍 및 행동 장애(ADHD 등)는 $0.863$으로 높은 성능을 보였다.
    - 예측 윈도우가 길수록(예: 180일) 예측 정확도가 높아지는 경향을 보였다.

2. **Doctor AI의 성능**:
    - 감별 진단(differential diagnosis)에서 $79\%$의 재현율(recall)을 달성하여 전문 의사와 대등한 수준의 성능을 보였다.
    - 다른 기관의 코딩 시스템이나 공개 데이터셋(MIMIC)에서도 일반화 능력이 입증되었다.

3. **종합 결과**: 모든 사례에서 딥러닝 기반 접근 방식이 전통적인 머신러닝 방식보다 우수한 성능을 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 기여
딥러닝은 EMR 데이터의 고차원성, 희소성, 노이즈 문제를 해결하기 위해 수동 특성 공학 없이도 데이터에서 직접 최적의 특징을 추출할 수 있음을 입증하였다. 특히 시계열 데이터 처리에 강한 LSTM/GRU와 패턴 인식에 강한 CNN, 표현 학습에 능한 Autoencoder 등이 각기 다른 임상적 목적에 맞게 활용되고 있다.

### 한계 및 비판적 해석
가장 심각한 병목 현상은 **해석 가능성(Interpretability)** 문제이다. 딥러닝 모델은 '블랙박스' 특성이 강해, 의료 전문가가 모델의 예측 근거를 이해하기 어렵다. 이는 실제 임상 현장에서 모델을 신뢰하고 채택하는 데 큰 장애물이 된다. 논문은 이를 해결하기 위해 EHDViz와 같은 실시간 시각화 대시보드의 필요성을 언급한다.

### 향후 전망
EMR 데이터뿐만 아니라 유전체 데이터(-omics), 환경 데이터(오염도 등), 웨어러블 센서 데이터와 같은 멀티모달(multimodal) 데이터의 통합이 이루어진다면, 보다 정밀한 다각적 질병 이해가 가능할 것이다. 특히 유전체 데이터의 희소성을 고려할 때, 딥러닝을 통한 EMR의 자동 처리는 현실적인 대안이 될 가능성이 높다.

## 📌 TL;DR

본 논문은 딥러닝이 EMR 데이터의 복잡성(희소성, 노이즈)을 극복하고 만성 질환 예측 및 환자 상태 모델링에서 전통적 머신러닝보다 뛰어난 성능을 보임을 체계적으로 리뷰한 연구이다. SDA, LSTM, GRU, CNN 등의 아키텍처가 질병 예측과 진단에 활용되고 있으며, 특히 Deep Patient와 Doctor AI 같은 모델이 높은 예측력을 입증하였다. 다만, 의료 현장 적용을 위해서는 모델의 '해석 가능성' 문제를 해결하는 것이 향후 연구의 핵심 과제이다.
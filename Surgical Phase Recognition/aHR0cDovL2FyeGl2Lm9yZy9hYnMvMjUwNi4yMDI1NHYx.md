# Recognizing Surgical Phases Anywhere: Few-Shot Test-time Adaptation and Task-graph Guided Refinement

Kun Yuan, Tingxuan Chen, Shi Li, Joel L. Lavanchy, Christian Heiliger, Ege Özsoy, Yiming Huang, Long Bai, Nassir Navab, Vinkle Srivastav, Hongliang Ren, and Nicolas Padoy (2025)

## 🧩 Problem to Solve

수술 워크플로우 인식(Surgical Workflow Understanding)은 환자의 안전을 높이고 수술실 내 의사소통을 최적화하는 데 필수적이다. 그러나 수술 환경은 병원마다 사용하는 프로토콜, 수술실 설정, 사용하는 도구, 그리고 환자의 해부학적 구조가 매우 다양하기 때문에, 한 기관에서 학습된 모델을 다른 기관에 적용했을 때 성능이 급격히 떨어지는 Domain Shift 문제가 발생한다.

기존의 해결책인 재학습(Retraining)은 각 새로운 임상 사이트마다 방대한 양의 데이터를 다시 어노테이션해야 하는 비용 문제가 있으며, 최근의 Few-shot learning 기법들 역시 수술 비디오 데이터의 특성상 샘플 수가 부족하여 과적합(Overfitting)이 발생하거나 테스트 시점의 분포 변화(Distribution Shift)에 취약하다는 한계가 있다. 따라서 본 논문의 목표는 최소한의 어노테이션만으로도 다양한 수술 환경에 신속하게 적응할 수 있는 경량화된 적응 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문은 Foundation Model을 다양한 수술 환경에 맞게 최적화하는 lightweight 적응 프레임워크인 **SurgicalPhaseAnywhere (SPA)**를 제안한다. SPA의 핵심 아이디어는 공간적(Spatial), 시간적(Temporal), 그리고 테스트 시점(Test-time)의 세 가지 차원에서 적응 전략을 통합하여 최소한의 감독(Minimal Supervision)만으로 일반화 성능을 극대화하는 것이다.

1. **공간적 적응(Spatial Adaptation):** 소량의 레이블링 된 이미지와 자연어 텍스트 설명을 사용하여 Foundation Model의 임베딩을 특정 기관의 수술 장면 및 단계에 정렬한다.
2. **시간적 적응(Temporal Adaptation):** 수술 프로토콜을 정의한 Task-graph를 기반으로 Diffusion Model을 학습시켜, 수술 단계 간의 전이 확률과 시간적 일관성을 강제한다.
3. **테스트 시점 적응(Test-time Adaptation, TTA):** 서로 다른 세 가지 예측 스트림 간의 상호 합의(Mutual Agreement)를 이용한 자기지도 학습을 통해, 추론 단계에서 발생하는 분포 변화에 실시간으로 대응한다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 대규모 데이터셋을 통한 지도 학습에 의존하였으며, 최근에는 SVL-Pretrain과 같은 대규모 비전-언어 데이터로 사전 학습된 Foundation Model을 활용하는 추세이다.

하지만 기존의 Few-shot transfer learning 방식들은 주로 이미지 기반의 공간적 적응에만 집중하여, 수술의 핵심인 시간적 흐름(Temporal dependency)을 무시하는 경향이 있다. 특히 수술 프로토콜은 병원마다 다르기 때문에, 단순한 이미지 분류 방식으로는 기관 간의 일반화 성능을 확보하기 어렵다. SPA는 이러한 한계를 극복하기 위해 기관별 Task-graph라는 시간적 사전 지식(Temporal prior)을 도입하여 기존 접근 방식과 차별화를 둔다.

## 🛠️ Methodology

SPA 프레임워크는 크게 세 가지 모듈로 구성된다.

### 1. Spatial Adaptation via Few-Shot Learning

Frozen 상태의 사전 학습된 비전 인코더 $\theta_v$와 텍스트 인코더 $\theta_t$를 사용하여 이미지 임베딩 $f_i$와 텍스트 설명 임베딩 $t_k$를 추출한다. 이후, 텍스트 기반의 선형 분류기 $P_f$를 학습시키는데, 이때 클래스 프로토타입 $w$와 학습 가능한 승수(Multiplier) $\alpha$를 사용하여 다음과 같이 확률을 계산한다.

$$p_{ik}(w,\alpha) = \frac{\exp (f_{i}^{\top}(w_k+\alpha_k t_k))}{\sum_{j=1}^{K}\exp (f_{i}^{\top}(w_j+\alpha_j t_j))}$$

여기서 텍스트 임베딩 $t_k$는 고정되며, $w$와 $\alpha$만을 최적화하여 매우 적은 양의 데이터로도 공간적 특징을 빠르게 학습한다.

### 2. Temporal Adaptation via Task-graph Guided Diffusion

초기 예측값의 노이즈를 제거하고 시간적 일관성을 부여하기 위해 Diffusion Model을 도입한다.

- **Task-graph 기반 데이터 생성:** 수술 단계 간의 유효한 전이를 정의한 그래프 $G=(V, E)$를 이용하여 합성 시퀀스를 생성한다. 다음 단계 $X_{l+1}$은 현재 단계 $X_l$에서 전이가 가능한 노드 집합에서 균등하게 샘플링되며, 단계별 최대 지속 시간 $L_{max}$를 초과하지 않아야 한다는 제약을 둔다.
  
$$P(X_{l+1}|X_l) = \begin{cases} U(v_j \in V | e_{ij} \in E) & \text{if } \Delta l_i < L_{i}^{max} \\ 0 & \text{otherwise} \end{cases}$$

- **Diffusion Process:** 합성된 시퀀스를 잠재 공간(Hidden state space) $H$로 매핑한 후, Forward process를 통해 점진적으로 가우시안 노이즈를 추가한다.
  $$q(H_t|H_{t-1}) = \mathcal{N}(H_t; \sqrt{1-\beta_t}H_{t-1}, \beta_t I)$$
  Reverse process에서는 학습된 파라미터 $\mu_\theta, \Sigma_\theta$를 통해 노이즈를 제거하며, 이를 통해 타겟 기관의 수술 프로토콜에 부합하는 매끄러운 단계 전이를 복원한다.

### 3. Test-time Adaptation (TTA) with Mutual Agreement

추론 단계에서 모델의 강건성을 높이기 위해 세 가지 예측 스트림을 생성한다: (1) 참조 이미지와의 유사도 기반 $S_{ref}$, (2) 비전-언어 매칭 기반 $S_{vl}$, (3) Few-shot 분류기 기반 $S_{fs}$.

이 세 스트림 간의 일관성을 높이기 위해 Contrastive Loss $L$을 사용하여 테스트 비디오에 맞게 모델을 미세 조정한다.
$$L = L_{mutual}(S_{ref}, S_{vl}) + L_{mutual}(S_{fs}, S_{ref})$$
$$L_{mutual}(A, B) = -\frac{1}{L} \sum_{l=1}^{L} \sum_{k=1}^{K} A_{l,k} \log B_{l,k}$$
이 과정을 통해 서로 다른 모달리티 간의 예측이 일치하도록 유도함으로써 테스트 시점의 분포 변화를 극복한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Cholec80(담낭 절제술), StrasBypass 및 BernBypass(위 우회술), Autolaparo(자궁 적출술).
- **사전 학습 모델:** PeskaVLP (Surgical vision-language foundation model).
- **비교 대상:** CLIP, Tip-Adapter-F, Linear Probe (LP), LP+text.

### 정량적 결과

실험 결과, SPA는 모든 벤치마크에서 기존 Few-shot 방법론들을 압도하는 성능을 보였다. 특히 32-shot 설정에서 Cholec80의 경우 F1 스코어를 기존 SOTA 대비 17.33% 향상시켰다. 놀라운 점은 BernBypass 데이터셋에서 32-shot으로 학습한 SPA(43.54% F1)가 전체 데이터를 사용한 Full-shot 모델(42.4% F1)보다 높은 성능을 기록했다는 것이다. 이는 약 300배 적은 데이터만으로도 효율적인 적응이 가능함을 시사한다.

### 정성적 결과 및 Ablation Study

- **노이즈 감소:** 정성적 분석 결과, 기존 방법들은 예측값이 파편화(Fragmentation)되는 경향이 심했으나, SPA는 TTA와 Diffusion 모델을 통해 단계 경계가 명확하고 시간적으로 일관된 예측을 생성하였다.
- **TTA의 효과:** TTA 적용 시 모든 데이터셋에서 F1 스코어가 유의미하게 상승하여, 테스트 시점의 분포 변화를 억제하는 데 효과적임을 입증하였다.
- **Task-graph의 전이성:** 해당 기관의 Task-graph(TG-In)를 사용했을 때 성능 향상이 뚜렷했으나, 타 기관의 그래프(TG-Cross)를 사용했을 때는 결과가 엇갈렸다. 이는 수술 프로토콜의 기관별 특수성이 매우 강함을 의미한다.

## 🧠 Insights & Discussion

본 연구의 강점은 단순히 모델의 파라미터를 튜닝하는 것에 그치지 않고, **'공간-시간-테스트 시점'**이라는 세 가지 계층적 적응 전략을 체계적으로 통합했다는 점이다. 특히 Diffusion Model을 이용해 수술 프로토콜이라는 도메인 지식을 소프트하게 제약 조건으로 부여한 점이 매우 효과적이었다.

다만, Task-graph의 전이성 실험에서 나타났듯이, 기관 간 프로토콜 차이가 클 경우 타 기관의 그래프를 그대로 사용하는 것이 오히려 독이 될 수 있다는 한계가 발견되었다. 이는 향후 연구에서 서로 다른 프로토콜 간의 공통 분모를 찾는 'Generic Task-graph'나 'Adaptive Graph' 연구가 필요함을 시사한다. 또한, 본 논문은 PeskaVLP라는 강력한 Foundation Model에 의존하고 있으므로, 기본 모델의 성능이 전체 시스템의 하한선을 결정한다는 가정이 깔려 있다.

## 📌 TL;DR

**SPA(SurgicalPhaseAnywhere)**는 최소한의 레이블(Few-shot)과 수술 프로토콜(Task-graph)만으로 다양한 병원 환경에 빠르게 적응하는 수술 단계 인식 프레임워크이다. 공간적 정렬, Diffusion 기반의 시간적 정제, 그리고 테스트 시점의 자기지도 학습(TTA)을 결합하여, 일부 케이스에서는 전체 데이터를 사용한 모델보다 더 뛰어난 성능을 보였다. 이 연구는 데이터 확보가 어려운 임상 현장에서 맞춤형 수술 AI 모델을 신속하게 배포할 수 있는 실질적인 방법론을 제시한다.

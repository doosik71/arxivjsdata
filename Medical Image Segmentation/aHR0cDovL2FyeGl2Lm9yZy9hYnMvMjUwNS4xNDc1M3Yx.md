# TransMedSeg: A Transferable Semantic Framework for Semi-Supervised Medical Image Segmentation

Mengzhu Wang, Jiao Li, Shanshan Wang, Long Lan, Huibin Tian, Liang Yang, Guoli Yang (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 제한된 라벨링 데이터로 학습해야 하는 Semi-Supervised Learning (SSL)의 한계를 해결하고자 한다. 의료 영상 데이터는 전문가의 어노테이션 비용이 매우 높기 때문에 SSL이 유망한 대안으로 제시되어 왔으나, 기존의 Consistency Regularization이나 Pseudo-labeling 기반 방식들은 다음과 같은 두 가지 근본적인 문제점을 가지고 있다.

첫째는 **해부학적 표현의 도메인 시프트(Domain Shift in Anatomical Representations)**이다. 스캐너 프로토콜, 환자군, 해부학적 구조의 차이로 인해 라벨링된 데이터와 라벨링되지 않은 데이터 사이의 분포 차이가 발생하며, 이는 모델의 일반화 성능을 저하시킨다. 둘째는 **소스 라벨 데이터에 대한 과도한 의존성**이다. 적은 양의 라벨링 데이터로 학습된 모델이 생성한 Pseudo-label을 통해 학습하는 과정에서, 소수 데이터에 내재된 편향(Bias)이 라벨링되지 않은 데이터로 전파되어 특히 희귀한 병변 등을 과소 분할(Under-segmentation)하는 문제가 발생한다.

따라서 본 연구의 목표는 서로 다른 임상 도메인 간의 전이 가능한 세만틱 관계를 활용하여, 도메인 시프트를 극복하고 해부학적 충실도(Anatomical Fidelity)를 유지하는 새로운 SSL 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Transferable Semantic Augmentation (TSA)** 모듈을 통해 도메인 간의 세만틱 정렬을 이루는 것이다. 단순히 데이터를 증강하는 것이 아니라, 특성 공간(Feature Space)에서 도메인 불변 세만틱을 정렬함으로써 특성 표현을 강화한다.

주요 설계 직관은 Student 네트워크(라벨링 데이터 학습)와 Teacher 네트워크(라벨링되지 않은 데이터 적응) 간의 상호작용을 통해 도메인 간의 차이를 명시적으로 모델링하는 것이다. 특히, 클래스별 통계량(평균과 공분산)을 이용하여 타겟 도메인의 특성을 반영한 특성 변환을 수행함으로써, 데이터 생성 없이도 암시적으로 세만틱 증강을 구현하였다. 또한, 계산 비용을 줄이기 위해 명시적인 샘플링 대신 이론적으로 유도된 상한 손실 함수(Upper-bound Loss)를 사용하여 최적화를 수행한다.

## 📎 Related Works

기존의 Semi-supervised Medical Image Segmentation (SSMIS) 연구들은 주로 Consistency Regularization, Pseudo-labeling, 그리고 Adversarial Training 전략을 사용하였다. 이러한 방식들은 라벨링되지 않은 데이터의 구조적, 문맥적 정보를 활용하여 모델의 일반화 능력을 높이려 노력했다.

본 논문은 특히 **GraphCL** (Graph-based Clustering for SSMIS)을 베이스라인으로 채택하고 있다. GraphCL은 그래프 신경망(GNN)을 통해 이미지 영역 간의 구조적 관계를 캡처하여 성능을 높였으나, 임상 환경에서 흔히 발생하는 도메인 시프트를 처리하는 메커니즘이 부족하다는 한계가 있다. TransMedSeg는 GraphCL의 구조적 이점을 유지하면서 TSA 모듈을 추가하여 도메인 간 전이 능력을 확보함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
TransMedSeg는 Teacher-Student 네트워크 구조를 기반으로 하며, Student 네트워크 $f_{stu}$는 라벨링된 소스 데이터 $D_s$에서 학습하고, Teacher 네트워크 $f_{tea}$는 라벨링되지 않은 타겟 데이터 $D_t$에 적응한다. 전체 손실 함수는 다음과 같이 정의된다.

$$L_{TransMedSeg} = L_{GraphCL} + \beta L_{tsa}$$

여기서 $L_{GraphCL}$은 기본 프레임워크의 손실이며, $L_{tsa}$는 제안된 전이 가능 세만틱 증강 손실이다. $\beta$는 증강의 강도를 조절하는 가중치이다.

### 2. Transferable Semantic Augmentation (TSA)
TSA는 소스 도메인의 특성을 타겟 도메인의 세만틱 방향으로 변환하여 도메인 간 간극을 메우는 모듈이다.

**특성 통계량 계산:**
각 클래스 $c$에 대해 Student 네트워크의 소스 특성 평균 $\mu_s^c$와 공분산 $\Sigma_s^c$를 계산한다. 동시에 Teacher 네트워크는 EMA(Exponential Moving Average) 업데이트를 통해 타겟 도메인의 통계량 $\mu_t^c$와 $\Sigma_t^c$를 추정한다.

**특성 증강 프로세스:**
도메인 간의 차이를 보정하기 위해 다음과 같은 다변량 정규분포에서 섭동 벡터(Perturbation vector) $z_c$를 샘플링한다.

$$z_c \sim \mathcal{N}(\Delta\mu_c, \Sigma_t^c)$$

여기서 $\Delta\mu_c = \mu_t^c - \mu_s^c$는 도메인 간의 체계적인 시프트(Systematic shift)를 캡처하며, $\Sigma_t^c$는 타겟 도메인의 클래스 내 변동성을 모델링하여 해부학적 구조의 일관성을 유지한다. 실제 증강된 특성 $\tilde{f}_{s,i}^c$는 다음과 같이 생성된다.

$$\tilde{f}_{s,i}^c = f_{s,i}^c + \delta, \quad \delta \sim \mathcal{N}(\alpha\Delta\mu_c, \alpha\Sigma_t^c)$$

### 3. 암시적 세만틱 데이터 증강 (ISDA) 및 손실 함수
명시적으로 수많은 샘플을 생성하여 학습하는 것은 메모리 비용이 매우 높다. 이를 해결하기 위해 본 논문은 **Implicit Semantic Data Augmentation (ISDA)** 방식을 도입한다.

수학적으로 $M \to \infty$일 때의 기대 손실을 분석하고, Jensen's inequality($\mathbb{E}[\log(X)] \le \log(\mathbb{E}[X])$)를 적용하여 계산 가능한 상한 손실 함수(Surrogate loss)를 유도하였다. 최종적으로 유도된 $L_{tsa}$는 다음과 같다.

$$L_{tsa} \approx \frac{1}{n_s} \sum_{i=1}^{n_s} \log \sum_{c=1}^{C} \mathbb{E} \left[ \exp(\Delta w_c^\top \tilde{f}_{s,i} + \Delta b_c) \right]$$

이 식을 통해 명시적인 샘플링 과정 없이도 무한한 증강 뷰(View)를 근사적으로 학습할 수 있으며, 계산 오버헤드를 거의 없이 도메인 정렬 효과를 얻을 수 있다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** ACDC (심장 MRI), Pancreas-NIH (췌장 CT), LA (좌심방 MRI).
- **평가 지표:** Dice Similarity Coefficient (Dice), Jaccard Index, 95% Hausdorff Distance (95HD), Average Surface Distance (ASD).
- **비교 대상:** V-Net, UA-MT, SASSNet, DTC, URPC, MC-Net, SS-Net, BCP, GraphCL 등 최신 SSL 방법론.

### 2. 정량적 결과
TransMedSeg는 모든 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였다.
- **LA 데이터셋:** 라벨 데이터 10% 사용 시 Dice 89.62%, Jaccard 81.31%를 기록하며 GraphCL을 능가하였다.
- **ACDC 데이터셋:** 10% 라벨 사용 시 Dice 89.96%, 95HD 1.61, ASD 0.64를 기록하였다. 특히 경계 지표인 95HD와 ASD에서 큰 개선을 보여, TSA가 해부학적 구조의 정밀한 묘사에 효과적임을 입증하였다.
- **Pancreas-NIH 데이터셋:** 췌장 분할의 높은 가변성에도 불구하고 Dice 83.06%를 달성하여 가장 높은 성능을 보였다.

### 3. 분석 및 시각화
- **Ablation Study:** $L_{tsa}$를 제거했을 때 모든 지표에서 성능 저하가 발생하였으며, 특히 데이터가 적은 상황(5% 라벨)에서 그 영향이 뚜렷하였다.
- **t-SNE 시각화:** $L_{tsa}$를 적용했을 때 특성 임베딩이 도메인 간에 잘 정렬되고 클래스별로 조밀하게 군집화되는 것을 확인하였다.
- **정성적 결과:** 시각화 결과, BCP나 GraphCL에 비해 장기의 경계면을 더 정확하게 추출하며, 특히 도메인 전이 영역에서의 오류가 현저히 적음을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 도메인 시프트 문제를 해결하기 위해 **공분산 인식 증강(Covariance-aware augmentation)**을 도입했다는 점이다. 단순히 평균을 맞추는 것이 아니라 타겟 도메인의 공분산을 활용함으로써, 장기의 모양이나 조직 경계와 같은 해부학적 타당성(Physiological plausibility)을 유지하며 특성을 변환할 수 있었다.

또한, 이론적인 분석을 통해 명시적 샘플링의 계산 비용 문제를 암시적 최적화(Implicit optimization)로 해결한 점이 돋보인다. 이는 실용적인 관점에서 추가적인 연산량 증가 없이 성능을 높일 수 있는 plug-in 모듈로서의 가능성을 보여준다.

다만, 논문에서 명시적으로 다루지 않은 부분은 다양한 모달리티 간의 전이(예: MRI $\to$ CT)에 대한 직접적인 실험이다. 현재는 각 데이터셋 내에서의 semi-supervised 설정에서의 도메인 시프트를 다루고 있다. 향후 연구에서 다기관(Multi-center) 데이터셋을 활용한 더 극단적인 도메인 시프트 상황에서의 검증이 필요할 것으로 보인다.

## 📌 TL;DR

TransMedSeg는 의료 영상 분할에서 라벨 데이터 부족으로 인한 편향과 도메인 시프트 문제를 해결하기 위해 **Transferable Semantic Augmentation (TSA)** 프레임워크를 제안한다. 이 연구는 Teacher-Student 구조와 클래스별 특성 통계량을 활용해 도메인 간 세만틱을 정렬하며, 수학적 상한 손실 함수를 유도하여 계산 효율성과 성능을 동시에 확보하였다. 결과적으로 ACDC, Pancreas-NIH, LA 등 주요 벤치마크에서 SOTA 성능을 기록하였으며, 특히 해부학적 경계 복원 능력이 탁월함을 입증하였다. 이는 향후 적은 라벨로 고성능의 의료 영상 분할 모델을 구축하는 연구에 중요한 기여를 할 것으로 보인다.
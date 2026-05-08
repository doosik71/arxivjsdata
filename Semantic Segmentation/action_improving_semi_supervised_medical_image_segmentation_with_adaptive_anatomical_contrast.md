# ACTION++: Improving Semi-supervised Medical Image Segmentation with Adaptive Anatomical Contrast

Chenyu You, Weicheng Dai, Yifei Min, Lawrence Staib, Jas Sekhon, and James S. Duncan

## 🧩 Problem to Solve

의료 데이터는 종종 심한 클래스 불균형(long-tail distribution)을 보여, 경계 영역이나 희귀 객체와 같은 소수 클래스 분류를 어렵게 만듭니다. 기존 반지도 학습(SSL) 방법론, 특히 대조 학습(CL)은 비지도 데이터에서는 상당한 개선을 보였지만, 클래스 분포가 불균형한 **레이블된 데이터**에서의 성능은 불분명합니다. 주요 문제는 다음과 같습니다:

1. **훈련 목표:** 기존 CL 방식은 비지도 대조 손실 설계에 초점을 맞추어 왔으며, long-tail 의료 데이터의 레이블된 부분에 대한 지도 대조 학습(supervised CL)은 충분히 연구되지 않았습니다.
2. **온도 스케줄러($\tau$):** 대조 손실에서 상수 온도 매개변수 $\tau$를 맹목적으로 사용하는 것은 long-tail 의료 데이터에 최적이지 않습니다. 큰 $\tau$는 그룹 수준 구별에 치우쳐 쉬운 피처에 편향될 수 있고, 작은 $\tau$는 픽셀 수준 구별을 강화하지만 그룹 수준 구별 능력을 저해할 수 있어, 동적인 $\tau$ 조절이 필요합니다.

## ✨ Key Contributions

* **ACTION++ 프레임워크 제안:** 적응형 해부학적 대조(adaptive anatomical contrast)를 활용하여 반지도 의료 영상 분할 성능을 개선한 프레임워크를 제안합니다.
* **Supervised Adaptive Anatomical Contrastive Learning (SAACL) 개발:**
  * 피처 공간이 지배적인 head 클래스에 편향되는 것을 방지하기 위해, 임베딩 공간에 균일하게 분포된 최적의 클래스 중심을 **오프라인으로 사전 계산**합니다.
  * 온라인 대조 매칭 학습을 통해 다양한 클래스 피처가 이들 고유하고 균일하게 분포된 클래스 중심에 적응적으로 매칭되도록 유도합니다.
* **Anatomical-aware Temperature Scheduler (ATS) 제안:**
  * 대조 손실에서 상수 $\tau$가 최적이지 않음을 발견하고, 간단한 코사인 스케줄을 통해 **동적 $\tau$**를 사용하여 다수 클래스와 소수 클래스 간의 더 나은 분리 능력을 달성합니다.
* **최첨단 성능 달성:** ACDC 및 LA 벤치마크 데이터셋에서 두 가지 반지도 설정(레이블 비율) 모두에서 기존 방법론들을 능가하는 최첨단 성능을 입증했습니다.
* **이론적 분석:** 적응형 해부학적 대조의 성능을 이론적으로 분석하여 레이블 효율성 측면에서의 우수성을 확인했습니다.

## 📎 Related Works

본 연구는 반지도 학습(SSL) 및 대조 학습(CL) 분야의 최신 연구들을 기반으로 합니다.

* **SSL 접근 방식:** 적대적 학습(adversarial training) [32,39,16,38], 딥 코-트레이닝(deep co-training) [23,43], 평균 교사(mean teacher) 방식 [27,42,14,13,15,7,41,34], 멀티태스크 학습(multi-task learning) [19,11,22,37,35], 대조 학습(contrastive learning) [2,29,40,33,24,36] 등이 언급됩니다. 특히, CL [2,29,36]이 파라미터 및 어노테이션 비용 증가 없이 의료 영상 분할 성능을 향상시키는 데 효과적임이 강조됩니다.
* **Long-tail 문제:** 의료 영상 데이터는 Zipfian 분포 [44]를 따르는 long-tailed (혹은 heavy-tailed) 클래스 분포를 가지며, 이러한 불균형 시나리오가 CL 방법론에 도전적임 [18]이 지적됩니다.
* **온도 매개변수($\tau$):** 대조 손실에서 $\tau$가 유용한 표현 학습에 중요한 역할을 함 [5,4]이 알려져 있으며, 큰 $\tau$는 그룹 수준 패턴을, 작은 $\tau$는 픽셀 수준 구별을 강조함 [28,25]이 언급됩니다. 또한, 그룹 수준 구별이 인스턴스 구별 능력을 감소시키고 "쉬운" 피처에 편향될 수 있음 [25]을 지적합니다.
* **기반 연구:** ACTION [36] 파이프라인을 백본 모델로 사용하며, [17]의 아이디어에서 영감을 받아 SAACL을 개발했습니다. [12]는 변화하는 $\tau$가 등방성 표현 공간을 유도함을 보여주었습니다.

## 🛠️ Methodology

ACTION++는 기존 ACTION [36] 파이프라인을 개선하며, student-teacher 프레임워크를 기반으로 합니다.

1. **전역 및 지역 사전 학습 (Global and Local Pre-training):**
    * ACTION [36]의 방식을 따릅니다.
    * 비지도 입력 스캔으로부터 증강된 뷰($x_1, x_2$)와 무작위로 샘플링된 뷰($x_3$)를 생성합니다.
    * 학생-교사 네트워크($F_s, F_t$)를 통해 전역 및 지역 임베딩($v_g, v_l$)을 추출합니다.
    * 뷰 간의 관계형 유사성을 SoftMax 함수로 처리한 후, Kullback-Leibler divergence를 이용하여 비지도 인스턴스 구별 손실 $L_{inst} = \text{KL}(u_s || u_t)$를 최소화합니다.
    * 최종 사전 학습 목표는 전역 및 지역 $L_{inst}$와 지도 분할 손실 $L_{sup}$ (Dice loss와 Cross-entropy loss의 조합)의 합입니다.

2. **해부학적 대조 미세 조정 (Anatomical Contrast Fine-tuning):**
    * 사전 학습된 모델이 long-tail 비지도 데이터에 취약한 문제를 완화하기 위해 ACTION [36]의 anatomical contrast 미세 조정을 적용합니다.
    * 추가적인 표현 헤드 $\phi$를 사용하여 조밀한 표현을 제공하고, AnCo 손실 $L_{anco}$를 사용하여 쿼리($r_q$)를 긍정 키($r_c^+_k$)에 가깝게 당기고 부정 키($r_c^-_k$)로부터 밀어냅니다.
    * 미세 조정 목표는 비지도 $L_{anco}$, 비지도 Cross-entropy 손실 $L_{unsup}$, 지도 분할 손실 $L_{sup}$의 합입니다.

3. **Supervised Adaptive Anatomical Contrastive Learning (SAACL):** (레이블된 데이터의 long-tail 문제 해결)
    * **해부학적 중심 사전 계산 (Anatomical Center Pre-computation):**
        * $d$-차원 공간의 단위 구 $S^{d-1}$에서 K개의 최적 클래스 중심 $\{\psi_c\}^K_{c=1}$을 오프라인으로 사전 계산합니다.
        * 균일성 손실 $L_{unif}(\{\psi_c\}^K_{c=1}) = \sum^K_{c=1} \log \left( \sum^{K}_{c'=1} \exp(\psi_c \cdot \psi_{c'} / \tau) \right)$를 최소화하여 클래스 간의 좋은 분리도와 균일성을 유도합니다. 이 단계는 훈련 데이터가 필요 없습니다.
    * **적응적 할당 (Adaptive Allocation):**
        * 사전 계산된 클래스 중심을 각 클래스에 적응적으로 할당합니다.
        * K-평균 알고리즘에서 영감을 받아, 현재 배치에서 각 클래스의 경험적 평균과 사전 계산된 중심 간의 거리를 최소화하는 할당 $\pi^*$를 반복적으로 탐색합니다.
    * **적응적 해부학적 대조 (Adaptive Anatomical Contrast):**
        * 각 클래스의 샘플 피처 표현이 해당 사전 계산된 클래스 중심 주위에 군집되도록 유도하기 위해 지도 대조 손실 $L_{aaco}$를 사용합니다.
        $$L_{aaco} = -\frac{1}{n} \sum^n_{i=1} \left( \sum_{\phi^+_i} \log \frac{\exp(\phi_i \cdot \phi^+_i / \tau_{sa})}{\sum_{\phi_j} \exp(\phi_i \cdot \phi_j / \tau_{sa})} + \lambda_a \log \frac{\exp(\phi_i \cdot \nu_i / \tau_{sa})}{\sum_{\phi_j} \exp(\phi_i \cdot \phi_j / \tau_{sa})} \right)$$
        여기서 $\nu_i = \psi^*_{\pi^*(y_i)}$는 픽셀 $i$의 클래스 $y_i$에 할당된 사전 계산된 중심입니다.

4. **Anatomical-aware Temperature Scheduler (ATS):**
    * 모든 지도 및 비지도 대조 손실에서 사용됩니다.
    * 온도 매개변수 $\tau$는 훈련 반복 횟수 $t$에 따라 $\tau_t = \tau^- + 0.5(1 + \cos(2\pi t/T))(\tau^+ - \tau^-)$와 같은 코사인 스케줄로 변화합니다.

## 📊 Results

ACTION++는 ACDC 및 LA 데이터셋에서 두 가지 반지도 설정(5%, 10% 레이블 비율) 모두에서 최첨단 성능을 달성했습니다.

* **LA 데이터셋 (표 1):** 4% 및 8% 레이블 설정에서 기존 모든 SSL 방법론 대비 가장 우수한 성능을 보였습니다. 예를 들어, 8% 레이블 설정에서 Dice 계수 89.9%, ASD(Average Surface Distance) 1.74 voxel을 기록했습니다.
* **ACDC 데이터셋 (표 2):** 3% 및 7% 레이블 설정에서 Dice 및 ASD 측면에서 최고 분할 성능을 달성했으며, 특히 RV(Right Ventricle) 및 Myo(Myocardium)와 같은 작고 도전적인 영역에서도 정확한 분할 경계를 제공했습니다.
* **정성적 결과 (그림 3, 4):** 시각화 결과는 ACTION++가 ACDC 및 LA 모두에서 더 정확하고 선명한 객체 경계를 생성함을 보여주었습니다.
* **Ablation Study (표 3, 4, 5, 6):**
  * **SAACL의 효과 (표 3):** SAACL의 '적응적 할당(adaptive allocation)' 구성 요소가 다른 SAACL 변형 및 기존 KCL, CB-KCL보다 훨씬 뛰어난 성능을 보였습니다.
  * **ATS 및 SAACL의 결합 효과 (표 5):** 사전 학습 및 미세 조정 단계에서 ATS와 SAACL을 모두 적용했을 때 최적의 성능을 달성했습니다.
  * **하이퍼파라미터 영향 (표 4, 6):** 온도 경계($\tau^-$ = 0.1, $\tau^+$ = 1.0), 코사인 스케줄 주기(T/#iterations = 1.0), 그리고 $\lambda_a$ = 0.2 등의 설정이 최적 성능에 기여함을 확인했습니다.

## 🧠 Insights & Discussion

* **함의:** ACTION++는 기존 반지도 학습 방법들이 주로 비지도 데이터의 클래스 불균형에 초점을 맞추는 것과 달리, **레이블된 데이터** 내의 클래스 불균형 문제를 명시적으로 해결하는 것이 중요함을 입증했습니다. 동적 온도 스케줄러와 오프라인으로 계산된 균일한 클래스 중심을 활용하여, 모델이 다수 클래스와 소수 클래스 모두에 대해 균형 잡힌 피처 공간을 학습하도록 유도합니다. 이는 long-tail 의료 데이터에서 일반화 능력을 향상시키고, 소수 클래스의 분할 정확도를 크게 개선하는 결과를 가져왔습니다. 이론적 분석을 통해 제안 방법이 낮은 오류율과 높은 클래스 발산(class divergence)을 유도하여 레이블 효율성을 높임을 보였습니다.
* **한계 및 향후 연구:** 현재 ACDC 및 LA 심장 데이터셋에 초점을 맞추고 있으므로, 향후에는 더 다양한 해부학적 구조와 더 많은 전경 레이블을 포함하는 CT/MRI 데이터셋에 대한 광범위한 검증이 필요합니다. 또한, t-SNE와 같은 시각화 기법을 활용하여 학습된 피처 표현 공간을 더 심층적으로 분석할 계획입니다.

## 📌 TL;DR

**문제:** 의료 영상 분할은 레이블/비지도 데이터 모두에서 클래스 불균형(long-tail) 문제를 겪고, 기존 대조 학습은 이에 취약하며 상수 온도($\tau$)가 최적이지 않습니다.
**제안 방법:** ACTION++는 **Supervised Adaptive Anatomical Contrastive Learning (SAACL)**과 **Anatomical-aware Temperature Scheduler (ATS)**를 도입합니다. SAACL은 오프라인에서 균일하게 분포된 최적 클래스 중심을 사전 계산하여 피처 매칭에 사용하며, ATS는 코사인 스케줄을 통해 $\tau$를 동적으로 조절하여 다수/소수 클래스 간의 분리를 강화합니다.
**주요 결과:** ACDC 및 LA 데이터셋에서 최첨단 성능을 달성했으며, 제안된 SAACL과 ATS가 long-tail 의료 영상 분할의 정확도와 레이블 효율성을 크게 향상시킴을 실험적, 이론적으로 입증했습니다.

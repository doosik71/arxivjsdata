# GlobalTrack: A Simple and Strong Baseline for Long-term Tracking
Lianghua Huang, Xin Zhao, Kaiqi Huang

## 🧩 Problem to Solve
기존 시각 추적기들은 대상의 위치와 스케일이 시간적으로 부드럽게 변화한다는 "시간적 일관성" 가정을 강하게 따르며, 주로 작은 영역 내에서 대상을 탐색합니다. 이러한 가정은 갑작스러운 움직임, 대상의 일시적인 부재, 또는 추적 실패와 같은 실제 환경의 도전 과제에 직면했을 때 추적 실패와 누적 오류로 이어지는 한계를 가집니다. 이 논문은 이러한 간극을 해소하고, 대상을 전체 이미지 영역에서 다중 스케일로 검색하여 장기 추적의 강건한 베이스라인을 제공하는 것을 목표로 합니다.

## ✨ Key Contributions
*   **순수 전역 인스턴스 검색 기반 추적기 제안:** 대상의 위치 및 스케일 변화에 대한 시간적 일관성 가정을 하지 않는 GlobalTrack을 제안하여 누적 오류 없이 장기 추적을 가능하게 합니다.
*   **쿼리 기반 RPN (QG-RPN) 및 RCNN (QG-RCNN) 개발:** 2단계 객체 탐지기(Faster-RCNN)에서 영감을 받아, 쿼리 정보를 활용하여 특정 대상에 특화된 제안(proposals)을 생성하고 분류 및 바운딩 박스 정제를 수행하는 모듈을 설계했습니다.
*   **교차 쿼리 손실(Cross-query Loss) 도입:** 이미지 내에 여러 인스턴스가 공존할 때, 다양한 쿼리에 대한 손실을 평균화하여 방해물에 대한 모델의 강건성을 향상시켰습니다.
*   **복잡한 후처리 없는 강력한 성능:** 온라인 학습, 위치/스케일 변화 페널티, 스케일 스무딩, 궤적 정제 등 복잡한 후처리 없이도 최첨단 추적기에 필적하거나 능가하는 성능을 달성했습니다.
*   **장기 추적 벤치마크에서의 우수성 입증:** LaSOT, TLP, OxUvA, TrackingNet 등 4가지 대규모 추적 벤치마크에서 기존 SOTA(State-of-the-Art) 추적기 대비 우수한 성능을 보여주었습니다.

## 📎 Related Works
*   **장기 추적기 (Long-term Trackers):**
    *   **TLD (Kalal et al. 2012), SPL (Supancic and Ramanan 2013), LCT (Ma et al. 2015), EBT (Zhu et al. 2016), DaSiam$_{\text{LT}}$ (Zhu et al. 2018):** 추적 실패 시 전체 이미지 검색을 수행하거나 검색 영역을 확장하는 접근 방식을 사용합니다. GlobalTrack은 이들과 유사하게 전체 이미지 검색이 가능하지만, 온라인 학습이나 시간적 일관성 제약을 사용하지 않아 누적 오류를 피합니다.
*   **추적 프레임워크 (Tracking Frameworks):**
    *   **SiamRPN (Li et al. 2018), ATOM (Danelljan et al. 2019):** 쿼리 및 검색 이미지 특징 추출을 위해 공유 백본을 사용하고 상관관계를 통해 관계를 인코딩한다는 점에서 GlobalTrack과 유사합니다. 그러나 이들은 주로 로컬 검색을 수행하며 복잡한 후처리를 필요로 하는 반면, GlobalTrack은 전역 검색을 수행하고 Faster-RCNN의 RPN 및 RCNN 헤드를 재사용한다는 점에서 차이가 있습니다.

## 🛠️ Methodology
GlobalTrack은 Faster-RCNN에서 영감을 받은 2단계 객체 탐지기 기반의 순수 전역 인스턴스 검색 추적기입니다.

1.  **쿼리 기반 특징 변조 (Query-Guided Feature Modulation):**
    *   **백본 (Backbone):** ResNet-50을 백본으로 사용하여 쿼리 이미지($I_z$)와 검색 이미지($I_x$)에서 특징 맵을 추출합니다.
    *   **QG-RPN (Query-Guided RPN):** 쿼리 인스턴스 ROI 특징 $z \in \mathbb{R}^{k \times k \times c}$를 컨볼루션 커널 $f_z(z)$로 변환하고, 이를 검색 이미지 특징 $f_x(x)$에 컨볼루션 연산($\otimes$)하여 쿼리 기반 특징 $\hat{x}$를 생성합니다.
        $$ \hat{x} = g_{\text{qg-rpn}}(z,x) = f_{\text{out}}(f_x(x) \otimes f_z(z)) $$
        생성된 $\hat{x}$는 QG-RPN의 분류 및 위치 예측을 위한 입력으로 사용됩니다.
    *   **QG-RCNN (Query-Guided RCNN):** QG-RPN에서 생성된 각 제안(proposal)의 ROI 특징 $x_i \in \mathbb{R}^{k \times k \times c}$와 쿼리 특징 $z$ 간의 상관관계를 Hadamard 곱($\odot$)을 통해 인코딩하여 변조된 특징 $\hat{x}_i$를 생성합니다.
        $$ \hat{x}_i = g_{\text{qg-rcnn}}(z,x_i) = h_{\text{out}}(h_x(x_i) \odot h_z(z)) $$
        변조된 특징 $\hat{x}_i$는 QG-RCNN의 분류 및 바운딩 박스 정제를 위한 입력으로 사용됩니다.

2.  **손실 함수 (Loss Function):**
    *   **QG-RPN 손실 ($L_{\text{qgrpn}}$):** Faster-RCNN의 RPN과 동일하게 이진 교차 엔트로피($L_{\text{cls}}$) 및 Smooth L1($L_{\text{loc}}$) 손실을 사용하여 제안의 분류 및 위치를 학습합니다.
        $$ L_{\text{qgrpn}}(z,x) = L_{\text{rpn}}(\hat{x}) = \frac{1}{N_{\text{cls}}} \sum_i L_{\text{cls}}(p_i,p^*_i) + \lambda \frac{1}{N_{\text{loc}}} \sum_i p^*_i L_{\text{loc}}(s_i,s^*_i) $$
    *   **QG-RCNN 손실 ($L_{\text{qgrcnn}}$):** QG-RPN과 유사하게 이진 교차 엔트로피 및 Smooth L1 손실을 사용하여 최종 예측을 최적화합니다.
        $$ L_{\text{qgrcnn}}(z,x) = \frac{1}{N_{\text{prop}}} \sum_i L_{\text{rcnn}}(\hat{x}_i) $$
    *   **교차 쿼리 손실 (Cross-query Loss, $L_{\text{cql}}$):** 동일 이미지에 공존하는 $M$개의 인스턴스에 대해 각각 쿼리-검색 이미지 쌍을 구성하고, 이들 각각의 $L_{\text{qgrpn}}$ 및 $L_{\text{qgrcnn}}$ 손실을 평균화하여 모델이 쿼리와 예측 결과 간의 강력한 의존성을 학습하도록 강제합니다.
        $$ L_{\text{cql}} = \frac{1}{M} \sum_{k=1}^M L(z_k,x) $$
        여기서 $L(z_k,x) = L_{\text{qgrpn}}(z_k,x) + L_{\text{qgrcnn}}(z_k,x)$입니다.

3.  **학습 및 추적 (Training & Tracking):**
    *   **오프라인 학습:** COCO, GOT-10k, LaSOT 데이터셋에서 무작위로 프레임 쌍을 샘플링하여 훈련합니다. 각 프레임 쌍에서 공존하는 여러 인스턴스를 쿼리로 사용하여 교차 쿼리 손실을 계산하고 SGD로 모델을 최적화합니다.
    *   **온라인 추적:** 첫 프레임에서 사용자 지정된 주석으로 쿼리를 초기화하고, 이후 추적 과정에서 쿼리를 고정(업데이트 없음)합니다. 각 새 프레임에서 QG-RPN과 QG-RCNN을 통해 예측을 수행하고, QG-RCNN의 Top-1 예측을 추적 결과로 바로 사용합니다. 추가적인 후처리(예: 궤적 스무딩)는 없습니다.

## 📊 Results
GlobalTrack은 4가지 대규모 추적 벤치마크에서 최첨단 추적기들과 비교하여 매우 인상적인 성능을 보여주었습니다.

*   **LaSOT (장기 추적):**
    *   AUC: 52.1% (SiamRPN++ 49.6%, ATOM 51.4% 대비 우수)
    *   Precision: 52.7%, Success: 52.1% (ATOM 대비 Precision 2.2%p, Success 0.7%p 향상)
    *   SPLT, DaSiam$_{\text{LT}}$ 등 최신 장기 추적기보다 큰 폭으로 우수합니다.
*   **TLP (초장기 추적):**
    *   Success Rate (SR@0.5): 63.8% (이전 최고 SPLT 52.7% 대비 11.1%p 절대 이득)
    *   매우 긴 비디오 추적에서 기존 알고리즘 대비 상당한 우위를 입증했습니다.
*   **TrackingNet (일반 추적):**
    *   Success (AUC): 70.4% (SiamRPN++ 73.3%, ATOM 70.3%와 비교할 만한 성능)
    *   대규모 테스트 데이터에 대한 일반화 능력을 보여주었습니다.
*   **OxUvA (장기 추적, 부재 예측 포함):**
    *   MaxGM (Maximum Geometric Mean): 60.3% (SiamFC+R 45.4% 대비 14.9%p 절대 이득)
    *   대상 부재/존재 예측 기능에 대한 강건성을 보여주었습니다 (Top-1 스코어 임계값 $\tau=0.84$로 예측).

**구성 요소별 분석:**
*   **QG-RPN 성능:** 일반 RPN(11.3%) 및 GA-RPN(12.9%) 대비 AR@1에서 67.1%로 압도적으로 높은 리콜을 보였습니다. 소수의 제안만으로도 높은 리콜을 달성하여 효율성을 입증했습니다.
*   **QG-RPN vs. QG-RCNN:** QG-RCNN은 AR@1(Top-1 정확도)에서 76.6%로 QG-RPN(67.1%)보다 높았지만, 제안 수가 많아질수록 QG-RPN이 더 높은 평균 리콜을 달성했습니다. 이는 각 모듈의 정확도/리콜에 대한 다른 선호도를 보여줍니다.
*   **교차 쿼리 손실의 영향:** 단일 쿼리 손실로 훈련된 모델 대비 모든 지표(Precision, Normalized Precision, Success)에서 2.6%~4.2% 향상되어 제안된 손실 함수의 강건성을 입증했습니다.

## 🧠 Insights & Discussion
*   **"시간적 일관성" 가정의 불필요성:** GlobalTrack은 기존 추적기의 근본적인 문제점인 시간적 일관성 가정에서 벗어나, 전역 인스턴스 검색을 통해 대상의 갑작스러운 상태 변화나 일시적 부재에도 강건하게 대응합니다. 이는 누적 오류를 방지하고 장기 추적 시나리오에서 큰 이점을 제공합니다.
*   **단순함 속의 강건함:** 복잡한 온라인 학습이나 후처리 없이도 최첨단 성능을 달성했다는 점은 이 방법론의 단순성과 효과성을 강조합니다. 이는 새로운 장기 추적 연구를 위한 강력한 베이스라인 역할을 할 수 있습니다.
*   **탐지-추적 간의 간극 해소:** 객체 탐지 모델을 기반으로 한 접근 방식은 탐지 모델의 강력한 특징 추출 및 위치 파악 능력을 추적에 효과적으로 활용하는 방향성을 제시합니다.
*   **제한점 및 향후 연구:** 현재 모델은 매우 단순하게 설계되었으며, 궤적 스무딩이나 더 적응적인 추적 모델과 같은 추가적인 개선이 성능 향상에 기여할 수 있음을 시사합니다. 이는 향후 연구의 방향이 될 수 있습니다.

## 📌 TL;DR
GlobalTrack은 대상의 위치 및 스케일 변화에 대한 시간적 일관성 가정을 제거한 순수 전역 인스턴스 검색 기반의 장기 추적기입니다. Faster-RCNN에서 영감을 받아 쿼리 기반 RPN(QG-RPN)과 RCNN(QG-RCNN)을 활용하여 전체 이미지에서 대상을 탐지하며, 교차 쿼리 손실을 통해 방해물에 대한 강건성을 높였습니다. 온라인 학습이나 복잡한 후처리 없이도 LaSOT, TLP 등 대규모 장기 추적 벤치마크에서 최첨단 성능을 능가하며, 이전 프레임의 추적 실패에 영향을 받지 않는 누적 오류 없는 추적 성능을 제공합니다.
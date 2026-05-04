# GUSOT: Green and Unsupervised Single Object Tracking for Long Video Sequences

Zhiruo Zhou, Hongyu Fu, Suya You, and C.-C. Jay Kuo (2022)

## 🧩 Problem to Solve

본 논문은 자원 제한적인 환경(Resource-constrained environment)에서 긴 비디오 시퀀스에 대한 단일 객체 추적(Single Object Tracking, SOT)을 수행하는 것을 목표로 한다. 최근의 지도 학습(Supervised) 및 비지도 학습(Unsupervised) 기반의 딥러닝 추적기들은 높은 추적 성능을 보이지만, 동시에 막대한 계산 복잡도와 메모리 비용을 요구한다. 이로 인해 드론, 자율주행 자동차, 모바일 기기와 같은 엣지 컴퓨팅 플랫폼(Edge computing platforms)에 배포하는 데 어려움이 있다.

또한, 기존의 많은 추적기들은 단기 추적(Short-term tracking)에 최적화되어 있어, 긴 비디오 시퀀스에서 빈번하게 발생하는 객체 추적 손실(Tracking loss) 상황에서 자동으로 복구하는 능력이 부족하다는 문제가 있다. 따라서 본 연구는 연산 효율성이 높은 'Green' 추적기이면서, 비지도 학습 기반으로 작동하고, 긴 시퀀스에서도 강건하게 객체를 추적 및 복구할 수 있는 GUSOT를 제안한다.

## ✨ Key Contributions

GUSOT의 핵심 아이디어는 단기 추적 성능이 검증된 경량 baseline 추적기인 UHP-SOT++를 기반으로 하되, 긴 비디오 시퀀스에서 발생하는 한계를 극복하기 위해 두 가지 새로운 모듈을 추가하는 것이다.

1.  **Lost Object Recovery (객체 손실 복구):** 배경 모션 추정 및 보상을 통해 객체가 사라졌을 때 다시 찾을 수 있는 motion proposal을 생성하고, 신뢰할 수 있는 템플릿과의 유사도를 비교하여 최적의 위치를 복구한다.
2.  **Color-Saliency-Based Shape Proposal (색상 돌출도 기반 형상 제안):** 저비용 세그멘테이션 기술과 색상 돌출도(Color Saliency) 분석을 통해 객체의 타이트한 Bounding Box 형상을 제안함으로써, 추적 중 발생하는 형상 변형에 유연하게 대응하고 템플릿 학습 시 노이즈를 줄인다.

## 📎 Related Works

본 논문은 추적기를 크게 세 가지 범주로 구분하여 설명한다.

-   **Supervised Deep Trackers:** 대량의 레이블링된 데이터를 사용하여 높은 정확도를 달성한다. 특히 Siamese-network 기반 추적기나 Transformer 기반 추적기가 대표적이다. 하지만 학습 데이터에 없는 객체에 대한 신뢰성 문제와 높은 연산 비용이 한계로 지적된다.
-   **Unsupervised Deep Trackers:** 레이블 없이 비디오 내의 순환 일관성(Cycle consistency) 등을 이용해 학습한다. 최근 ResNet-50과 같은 무거운 백본을 사용하여 성능을 높이고 있으나, 여전히 자원 제한적 환경에서 사용하기에는 무겁고 DCF 기반 추적기에 비해 성능 이득이 제한적인 경우가 많다.
-   **Unsupervised Conventional Trackers:** Discriminative Correlation Filters (DCF) 기반 추적기가 대표적이다. FFT(Fast Fourier Transform)를 사용하여 CPU에서도 효율적으로 동작하지만, 딥러닝 기반 추적기에 비해 정확도가 크게 떨어진다.

GUSOT는 이러한 기존 연구들의 한계를 극복하기 위해, DCF의 효율성과 비지도 학습의 유연성을 결합하고, 여기에 객체 복구 및 형상 최적화 메커니즘을 추가하여 성능과 효율성의 균형을 맞추고자 하였다.

## 🛠️ Methodology

GUSOT의 전체 파이프라인은 baseline인 UHP-SOT++가 제공하는 box proposal(빨간색 박스)과 새롭게 제안된 recovery 모듈의 motion proposal(파란색 박스), 그리고 shape proposal(노란색 박스)의 상호작용으로 구성된다.

### 1. Lost Object Recovery
객체를 놓쳤을 때 이를 효율적으로 찾기 위해 배경 모션 추정 및 보상(Background motion estimation and compensation) 기법을 사용한다.

-   **절차:** 프레임 $(t-1)$과 $t$ 사이의 희소하게 샘플링된 배경 돌출점들의 대응 관계를 통해 전역 모션 필드(Global motion field)를 추정한다. 이를 통해 모션 보상 프레임을 생성하고, 현재 프레임과의 차이를 계산하여 모션 잔차 맵(Motion residual map)을 도출한다. 이 맵에서 잔차가 가장 큰 영역을 기반으로 motion proposal을 생성한다.
-   **선택 기준:** baseline proposal($x_2$)과 motion proposal($x_1$) 중 더 나은 것을 선택하기 위해 '신뢰할 수 있는 템플릿' $f^*$와의 유사도를 측정한다.
    -   **상관 계수(Correlation Coefficient):** 특징 표현의 벡터 내적을 이용한다.
      $$s_1(f^*, x) = \frac{\langle f^*, x \rangle}{\|f^*\| \|x\|}$$
    -   **Chi-square 거리:** 색상 히스토그램 $v$의 유사도를 측정한다.
      $$s_2(f^*, x) = \sum_{i} \frac{(v_{f^*,i} - v_{x,i})^2}{v_{f^*,i} + v_{x,i}}$$
-   **결정:** $s_1$ 값이 더 크고 $s_2$ 값이 더 작을 때(더 유사할 때) motion proposal을 채택한다.

### 2. Color-Saliency-Based Shape Proposal
객체의 형상을 정밀하게 추정하기 위해 저비용 세그멘테이션을 수행한다.

-   **MRF 기반 세그멘테이션:** Markov Random Field (MRF) 최적화 프레임워크를 사용하여 픽셀별 이진 레이블 $l^p \in \{0, 1\}$을 할당한다.
  $$I^* = \arg \min_{I} \sum_{p} \rho(p, l^p) + \sum_{\{p,q\} \in \mathcal{N}} w_{pq} \|l^p - l^q\|$$
  여기서 $\rho(p, l^p)$는 가우시안 혼합 모델(GMM)에서의 음의 로그 우도이며, 두 번째 항은 인접 픽셀 간의 레이블 연속성을 보장한다.
-   **Color Saliency Score (CSS):** MRF의 초기화를 정확하게 하기 위해 색상 돌출도를 계산하여 foreground/background 샘플링 지점을 정한다.
  $$CSS(k_i) = \frac{\sum_{j \neq i} \exp \|k_i - k_j\|^2}{Z} (p_{in}(k_i) - p_{out}(k_i))$$
  여기서 $k_i$는 색상 키, $p_{in}$과 $p_{out}$은 각각 박스 내부와 외부의 색상 분포이다. CSS 값이 크면 foreground, 작으면 background로 판단하여 샘플링한다.
-   **Fallback 전략:** 객체가 너무 복잡하여 MRF 결과가 비정상적일 경우, 슈퍼픽셀(Superpixel) 세그멘테이션을 기반으로 형상을 제안하는 방식으로 전환한다.

### 3. Final Output Proposal
최종 예측 박스 $B^*$는 baseline proposal($B_b$), motion proposal($B_m$), shape proposal($B_s$)을 종합하여 결정한다. baseline의 유사도 점수가 높으면 그대로 유지하며, 낮을 경우 다음 식을 통해 최적의 $B_s$를 선택한다.
$$B^* = \arg \max_{B_s} \text{IoU}(B_s, B_b) + \text{IoU}(B_s, B_m)$$

## 📊 Results

### 실험 설정
-   **데이터셋:** LaSOT (280개의 긴 비디오 시퀀스, 약 685K 프레임)
-   **평가 지표:** 20-pixel 임계값에서의 거리 정밀도(Distance Precision, DP) 및 중첩 정밀도의 곡선 아래 면적(Area-Under-Curve, AUC)
-   **구현 세부사항:** baseline의 외형 점수가 0.2 미만일 때 세그멘테이션 모듈이 활성화되며, 세그멘테이션 패치 크기는 $48 \times 48$로 설정되었다.

### 주요 결과
-   **정량적 결과:** GUSOT는 DP 36.1%, AUC 36.8%를 기록하였다. 이는 baseline인 UHP-SOT++ 대비 DP는 약 10%, AUC는 약 12% 향상된 수치이다.
-   **비교 분석:**
    -   기존 DCF 기반 추적기(ECO, STRCF 등)보다 월등히 높은 성능을 보였다.
    -   사전 학습이 필요한 일부 비지도 딥러닝 추적기(LUDT, USOT)보다 우수한 성능을 나타냈다.
    -   지도 학습 기반의 딥러닝 추적기(SiamFC, ECO)보다 높은 성능을 보였으며, SiamRPN과의 격차를 상당히 좁혔다.
    -   다만, 방대한 양의 가상 레이블로 사전 학습된 ULAST보다는 성능이 낮았으나, ULAST는 ResNet-50 백본을 사용하여 엣지 기기 적용이 어렵다는 점이 명시되었다.
-   **속성별 평가:** 특히 빠른 움직임(Fast motion), 시야 이탈(Out-of-view), 시점 변화(Viewpoint change) 상황에서 성능 향상이 뚜렷했으며, 이는 제안된 복구 및 형상 제안 모듈의 효과임을 입증하였다.
-   **Ablation Study:** motion proposal과 shape proposal을 각각 추가했을 때 모두 성능 향상이 있었으며, 특히 긴 시퀀스 추적에서는 shape proposal의 기여도가 더 컸다. 또한, KCF, STRCF 등 다른 DCF 기반 추적기에 적용했을 때도 일관된 성능 향상을 보였다.

## 🧠 Insights & Discussion

본 논문은 딥러닝의 무거운 연산량 없이도 고성능 추적기를 구현할 수 있음을 보여주었다. 특히, 단순히 성능 수치에만 집중하지 않고 'Green'이라는 키워드를 통해 모바일 및 엣지 컴퓨팅 환경에서의 실용성을 강조한 점이 돋보인다.

**강점:**
-   **효율성:** 딥러닝 백본 없이 DCF와 전통적인 컴퓨터 비전 기법(MRF, Superpixel)을 결합하여 매우 가벼운 구조를 유지하면서도 경쟁력 있는 성능을 냈다.
-   **강건성:** 객체 손실 시 모션 잔차를 이용한 복구 메커니즘과 색상 돌출도 기반의 형상 최적화를 통해 긴 비디오 시퀀스의 고질적인 문제들을 효과적으로 해결하였다.

**한계 및 논의사항:**
-   **형상 변형 대응:** 속성 평가에서 변형(Deformation) 항목은 여전히 USOT와 같은 박스 회귀 신경망 기반 추적기가 더 우세하였다. 이는 전통적인 세그멘테이션 방식이 복잡한 비강체 변형(Non-rigid deformation)을 완전히 캡처하는 데 한계가 있음을 시사한다.
-   **사전 가정:** 배경 모션 보상이 유효하려면 배경의 움직임이 어느 정도 일관적이어야 한다는 가정이 전제되어 있다. 매우 불규칙한 배경 움직임이 존재하는 환경에서의 성능은 명확히 제시되지 않았다.

## 📌 TL;DR

GUSOT는 자원 제한적인 환경에서 긴 비디오를 추적하기 위한 **비지도 경량 추적기**이다. UHP-SOT++를 baseline으로 하여 **모션 잔차 기반의 객체 복구 모듈**과 **색상 돌출도 기반의 형상 제안 모듈**을 추가함으로써, 딥러닝 추적기에 육박하는 성능을 내면서도 연산 효율성을 극대화하였다. 이 연구는 특히 연산 능력이 낮은 모바일 및 엣지 디바이스에서 실시간 객체 추적을 구현하는 데 중요한 해결책을 제시한다.
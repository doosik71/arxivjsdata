# MobileInst: Video Instance Segmentation on the Mobile

Renhong Zhang, Tianheng Cheng, Shusheng Yang, Haoyi Jiang, Shuai Zhang, Jiancheng Lyu, Xin Li, Xiaowen Ying, Dashan Gao, Wenyu Liu, Xinggang Wang (2024)

## 🧩 Problem to Solve

본 논문은 모바일 기기에서의 Video Instance Segmentation (VIS) 수행 시 발생하는 효율성 문제를 해결하고자 한다. VIS는 비디오 시퀀스 내의 객체를 동시에 식별(Identify), 분할(Segment), 추적(Track)해야 하는 매우 도전적인 Edge AI 문제이다.

기존의 VIS 방법론들은 크게 두 가지 방향으로 나뉘지만, 모바일 기기에 적용하기에는 다음과 같은 한계가 존재한다.

1. **Offline 방법 (Clip-level):** 비디오를 클립 단위로 나누어 처리하며 풍부한 문맥 정보를 활용하지만, 모바일 기기의 제한된 연산 능력과 메모리 비용으로 인해 클립 단위 추론은 사실상 불가능하다.
2. **Online 방법 (Frame-level):** 프레임 단위로 예측을 수행하지만, 프레임 간 객체를 연관시키기 위해 NMS(Non-Maximum Suppression)와 같은 복잡한 휴리스틱(Heuristic) 절차를 필요로 하며, 이는 모바일 환경에서 매우 비효율적이다.

또한, 최신 Transformer 기반 모델들은 연산 부담이 너무 커서 모델 크기를 단순히 줄일 경우 성능 저하가 심각하게 발생한다. 따라서 본 연구의 목표는 모바일 기기에서 저지연(Low latency)과 높은 성능을 동시에 달성할 수 있는 경량화된 VIS 프레임워크인 MobileInst를 제안하는 것이다.

## ✨ Key Contributions

MobileInst의 핵심 아이디어는 **"경량화된 프레임별 분할 구조"**와 **"단순하지만 효과적인 시간적 모델링을 통한 객체 추적"**의 결합이다.

- **Query-based Dual Transformer Instance Decoder:** 무거운 6단계 디코더 대신, 전역 문맥을 파악하는 Global Decoder와 세부 공간 정보를 보완하는 Local Decoder의 2단계 구조를 제안하여 연산량을 획기적으로 줄였다.
- **Semantic-enhanced Mask Decoder:** Transformer 기반의 픽셀 디코더 대신, 반복적인 Top-down/Bottom-up 융합과 Semantic Enhancer를 사용하여 적은 비용으로 고해상도 마스크 특징을 생성한다.
- **Kernel Reuse & Association:** 객체 쿼리(Object Query)가 생성한 마스크 커널(Mask Kernel)이 인접 프레임 간에 시간적 일관성(Temporal Consistency)을 가진다는 점에 착안하여, 특정 프레임의 커널을 재사용하고 코사인 유사도(Cosine Similarity)로 연관시키는 단순한 추적 방식을 도입했다.
- **Temporal Query Passing:** 학습 단계에서 서로 다른 두 프레임 간의 쿼리를 전달하는 전략을 통해 추가적인 파라미터 없이 모델의 추적 능력을 강화했다.

## 📎 Related Works

### 1. Instance Segmentation

Mask R-CNN과 같은 2단계 방식이나 YOLACT, SparseInst 같은 실시간 방식들이 제안되었다. 최근에는 query-based detector(예: DETR, Mask2Former)가 좋은 성과를 내고 있으나, 여전히 모바일 기기에 배포하기에는 연산 부담과 복잡한 후처리 과정이 걸림돌이 되고 있다.

### 2. Video Instance Segmentation

- **Offline VIS:** VisTR 등이 대표적이며 클립 단위로 처리한다. 높은 성능을 보이지만 메모리 점유율이 높아 모바일 적용이 어렵다.
- **Online VIS:** 초기에는 CNN 기반 모델에 임베딩과 휴리스틱 매칭을 추가했으나 후처리가 복잡했다. 최근에는 IDOL, InsPro 등 Transformer 기반 모델들이 등장했으나, 이들은 Mask2Former나 Deformable DETR와 같이 거대한 모델에 의존하고 있어 모바일 기기의 역량을 초과한다.

### 3. Mobile Vision Transformers

MobileViT, TopFormer, SeaFormer 등 모바일 환경에 최적화된 백본 네트워크들이 제안되었다. 본 논문은 이러한 경량 백본을 활용함과 동시에, 백본 이후의 디코더 구조와 추적 메커니즘까지 모바일 최적화를 확장했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MobileInst는 **Mobile Transformer Backbone $\rightarrow$ Dual Transformer Instance Decoder $\rightarrow$ Semantic-enhanced Mask Decoder**의 순서로 구성된다.

### 2. Dual Transformer Instance Decoder

기존의 복잡한 디코더를 단순화하여 두 단계로 구성한다.

- **Global Instance Decoder:** 백본의 최상위 특징 $X_6$ (Global Features, $X_G$)를 사용하여 객체 쿼리 $Q$를 업데이트하고, 클래스 점수와 마스크 커널을 생성한다.
- **Local Instance Decoder:** 마스크 특징 $X_{mask}$를 Max Pooling을 통해 $\frac{1}{64} \times \frac{1}{64}$ 크기로 다운샘플링한 $X_L$ (Local Features)을 사용하여 쿼리를 정교화함으로써 공간적 세부 사항을 보완한다.

### 3. Semantic-enhanced Mask Decoder

고해상도 특징을 효율적으로 생성하기 위해 다음 과정을 거친다.

- **Multi-scale Fusion:** $\{X_3, X_4, X_5\}$ 특징들을 반복적인 Top-down 및 Bottom-up 방식으로 융합한다.
- **Semantic Enhancer (SE):** Global 특징 $X_6$를 활용하여 마스크 특징에 고수준의 의미 정보를 주입한다.
- **Mask Generation:** 생성된 커널 $K$와 마스크 특징 $X_{mask}$의 내적(Dot product)을 통해 최종 마스크 $M$을 얻는다.
$$M = K \cdot X_{mask}$$

### 4. Tracking: Kernel Reuse and Association

객체 쿼리는 시간적으로 일관된 특성을 가지므로, 이를 추적에 직접 활용한다.

- **Kernel Reuse:** 키프레임 $t$에서 생성된 커널 $K_t$를 이후 $T-1$개의 프레임 동안 재사용한다.
$$M_t = K_t \cdot X_{t}^{mask}, \quad M_{t+i} = K_t \cdot X_{t+i}^{mask} \quad (i \in \{0, \dots, T-1\})$$
- **Kernel Association:** 클립 간의 연결은 연속된 키프레임 간 커널의 코사인 유사도를 측정하여 단순하게 수행한다.

### 5. Temporal Training via Query Passing

추가 모듈 없이 추적 성능을 높이기 위해 학습 시 다음 전략을 사용한다.

- 비디오에서 두 프레임 $t$와 $t+\delta$를 무작위로 샘플링한다.
- 프레임 $t$의 Global Instance Decoder에서 생성된 쿼리 $Q_t^G$를 프레임 $t+\delta$의 Local Instance Decoder 입력으로 전달하여 마스크 $\tilde{M}_{t+\delta}$를 생성하고, 이를 정답 마스크와 비교하여 학습시킨다.

### 6. Loss Function

Bipartite matching을 통한 라벨 할당을 수행하며, 다음과 같은 손실 함수를 사용한다.
$$\mathcal{L} = \lambda_c \cdot \mathcal{L}_{cls} + \lambda_{mask} \cdot \mathcal{L}_{mask} + \lambda_{obj} \cdot \mathcal{L}_{obj}$$
여기서 $\mathcal{L}_{cls}$는 Focal loss, $\mathcal{L}_{mask}$는 Dice loss와 Binary Cross Entropy (BCE)의 조합, $\mathcal{L}_{obj}$는 IoU-aware objectness를 위한 BCE loss이다. 가중치는 $\lambda_c=2.0, \lambda_{mask}=2.0, \lambda_{obj}=1.0$으로 설정되었다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** COCO (이미지 인스턴스 분할), YouTube-VIS 2019 및 2021 (비디오 인스턴스 분할).
- **하드웨어:** Snapdragon 778G 모바일 플랫폼의 단일 CPU 코어.
- **가속화:** 혼합 정밀도(Mixed precision)나 양자화(Quantization) 없이 TNN 프레임워크를 사용하여 측정했다.

### 2. 주요 결과

- **COCO 데이터셋 (이미지):**
  - TopFormer 백본 사용 시 **31.2 mask AP**를 달성했으며, 추론 속도는 **433ms**이다.
  - 이는 기존 SOTA 방법 대비 지연 시간을 약 50% 감소시킨 결과이다.
  - Mask2Former와 유사한 AP를 보이면서 속도는 2배 더 빠르다.
- **YouTube-VIS 데이터셋 (비디오):**
  - **YouTube-VIS 2019:** 35.0 AP, 추론 속도 188ms.
  - **YouTube-VIS 2021:** 30.1 AP.
  - 동일 설정 하에서 CrossVIS나 SparseInst-VIS보다 더 높은 정확도와 빠른 속도를 보였다.

### 3. Ablation Study

- **Decoder 구조:** Global-Local 구조가 단일 디코더나 Local-Local 구조보다 비디오 추적 성능이 더 우수함을 확인했다.
- **Pooling 방식:** Local 특징 추출 시 Average Pooling보다 Max Pooling이 0.4 AP 더 높은 성능을 보였으며, 이는 불필요한 정보를 필터링하여 Global 특징와 더 좋은 상호보완 관계를 형성하기 때문으로 분석된다.
- **Temporal Training:** 제안된 쿼리 전달 학습 전략을 적용했을 때, 적용하지 않은 경우보다 약 0.8~1.3 AP의 성능 향상이 있었다.

## 🧠 Insights & Discussion

### 1. 강점

- **모바일 최적화의 선구적 접근:** VIS 작업을 위해 처음으로 모바일 CPU 환경을 타겟팅하여 설계된 프레임워크이다.
- **효율적인 추적 메커니즘:** 무거운 NMS나 복잡한 매칭 알고리즘 없이, 쿼리의 시간적 일관성과 커널 재사용만으로 실용적인 수준의 추적 성능을 확보했다.
- **유연한 속도-정확도 트레이드오프:** 커널 재사용 주기 $T$를 조절함으로써, 영상의 복잡도에 따라 추론 속도를 더욱 높일 수 있는 구조를 갖추고 있다.

### 2. 한계 및 비판적 해석

- **급격한 움직임에 취약:** 정성적 결과 분석에서 객체의 움직임이 매우 심한 장면에서는 경계선이 뭉툭해지거나(Coarse boundaries) 분할 품질이 떨어지는 현상이 관찰되었다. 이는 커널 재사용 방식이 프레임 간 변화가 적다는 가정을 전제로 하기 때문이다.
- **CPU 기반 측정의 한계:** Snapdragon 778G CPU에서 측정되었으나, 실제 모바일 기기의 NPU(Neural Processing Unit)나 GPU 가속을 활용했을 때의 성능 향상 폭에 대한 분석은 본문에서 명시적으로 다루어지지 않았다.

## 📌 TL;DR

본 논문은 모바일 기기에서 실시간 Video Instance Segmentation을 가능하게 하는 경량 프레임워크 **MobileInst**를 제안한다. **Global-Local Dual Transformer Decoder**와 **Semantic-enhanced Mask Decoder**를 통해 연산량을 줄였으며, **커널 재사용(Kernel Reuse)**과 **코사인 유사도 기반 연관(Association)** 방식을 통해 복잡한 후처리 없이 객체를 추적한다. 실험 결과, Snapdragon 778G CPU에서 기존 SOTA 대비 지연 시간을 50% 줄이면서도 경쟁력 있는 정확도를 달성했다. 이 연구는 리소스가 제한된 엣지 디바이스에서의 고수준 비전 인식 연구에 중요한 이정표가 될 것으로 보인다.

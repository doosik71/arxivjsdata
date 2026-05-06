# [CLS] Token is All You Need for Zero-Shot Semantic Segmentation

Letian Wu, Wenyao Zhang, Tengping Jiang, Wankou Yang, Xin Jin, Wenjun Zeng (2023)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Semantic Segmentation (ZS3) 문제를 해결하고자 한다. Semantic Segmentation의 일반적인 목표는 이미지의 모든 픽셀에 대해 카테고리를 예측하는 것이나, 기존의 Fully-supervised 방식은 막대한 양의 어노테이션 데이터에 의존하며, 학습 과정에서 보지 못한 새로운 클래스(Unseen categories)에 대해 일반화 성능이 떨어진다는 한계가 있다.

ZS3는 학습 단계에서 한 번도 본 적 없는 새로운 클래스를 추가적인 어노테이션 없이 분할하는 것을 목표로 한다. 이는 서로 다른 객체 간의 큰 간극과 활용 가능한 정보의 부족으로 인해 매우 도전적인 과제이다. 특히, 기존의 CLIP 기반 ZS3 방법론들은 단순히 이미지-텍스트 특징을 융합하여 디코더에 전달하거나(One-stage), 먼저 클래스 구분 없는 마스크를 생성한 후 CLIP을 분류기로 사용하는(Two-stage) 방식에 치중하여, CLIP이 가진 잠재적인 세그멘테이션 능력을 충분히 활용하지 못하고 있다. 따라서 본 연구의 목표는 CLIP의 텍스트 브랜치에서 추출한 전역적 의미 정보를 활용하여 시각적 인코더가 관심 영역(Region of Interest)에 더 집중하게 함으로써 ZS3 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **One-Way [CLS] token navigation**이다. 연구진은 CLIP의 텍스트 브랜치에서 생성되는 $[CLS]$ 토큰이 매우 강력한 세만틱 정보의 표현체이며, 이를 시각적 인코더의 가이드로 사용할 수 있다는 점에 주목하였다.

주요 기여 사항은 다음과 같다.

1. **ClsCLIP 제안**: CLIP의 시각적 인코더(ViT 기반)의 얕은 층(Shallow layers)에 있는 $[CLS]$ 토큰을 텍스트 브랜치의 $[CLS]$ 토큰으로 교체하여, 모델이 이미지의 초기 단계부터 특정 카테고리에 집중하도록 유도하는 단순하면서도 효과적인 ZS3 방법론을 제안한다.
2. **전역 카테고리 사전 정보의 조기 주입**: 텍스트-이미지 정렬이 잘 된 CLIP의 특성을 이용하여, 텍스트 측의 $[CLS]$ 토큰을 시각적 인코더에 조기에 임베딩함으로써 밀집 예측(Dense prediction) 성능을 향상시킨다.
3. **ClsCLIP+ 제안**: ZS3에서 흔히 발생하는 작은 객체(Tiny objects) 누락 문제를 해결하기 위해, YOLOv7과 같은 Region Proposal 생성기를 전처리에 도입하여 관심 영역을 먼저 확정한 후 세그멘테이션을 수행하는 Local zoom-in 전략을 제안한다.

## 📎 Related Works

본 논문은 크게 세 가지 관련 연구 분야를 다룬다.

**1. Pre-trained Vision Language Models (CLIP 등)**
CLIP은 대규모 이미지-텍스트 쌍을 대조 학습(Contrastive Learning)하여 강력한 일반화 능력을 갖춘 모델이다. 기존의 CLIP 기반 ZS3 방법들은 크게 두 가지 흐름으로 나뉜다. One-stage 방식은 두 모달리티의 특징을 융합하여 디코더로 보내는 형태이며, Two-stage 방식은 마스크 생성 후 CLIP을 분류기로 사용하는 형태이다. 하지만 이들은 CLIP의 텍스트-이미지 정렬 능력을 인코더 단계에서부터 능동적으로 활용하지 못한다는 한계가 있다.

**2. Generalized Semantic Segmentation**
Few-shot segmentation은 소수의 지원 세트(Support set)를 통해 새로운 클래스를 학습하지만, 여전히 최소 하나의 어노테이션이 필요하다. 반면 ZS3는 어떠한 픽셀 수준의 어노테이션 없이 보조적인 세만틱 정보(텍스트 등)만으로 예측을 수행해야 하므로, 본 논문은 CLIP의 제로샷 분류 능력을 세그멘테이션 영역으로 확장하려 한다.

**3. Prompt Learning**
최근 NLP와 CV 분야에서는 하드 프롬프트나 소프트 프롬프트를 통해 사전 학습 모델을 하위 작업에 맞게 가이드하는 연구가 활발하다. 본 논문은 텍스트 측의 $[CLS]$ 토큰을 일종의 보조 세만틱 프롬프트로 활용하여 시각적 인코더의 어텐션을 제어한다는 점에서 프롬프트 러닝의 관점을 채택하고 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 ClsCLIP 구조

ClsCLIP은 CLIP의 텍스트 인코더에서 얻은 $[CLS]$ 토큰을 시각적 인코더의 입력으로 주입하는 구조를 가진다.

1. **텍스트 인코더**: 보조 세만틱 단어 $A$를 입력받아 카테고리 $[CLS]$ 토큰 $T_{cls}$를 생성한다.
   $$T_{cls} = F_t(A), \quad T_{cls} \in \mathbb{R}^d$$
2. **시각적 인코더 (One-Way Navigation)**: ViT 기반의 시각적 인코더는 총 $N$개의 층으로 구성된다. 여기서 본 논문은 모든 층의 $[CLS]$ 토큰을 유지하는 대신, 특정 구간의 토큰을 $T_{cls}$로 교체한다.
   - $0 \le i < N_1$: 기존의 시각적 $[CLS]$ 토큰 $I_{cls}$를 사용한다.
   - $N_1 \le i < N_2$: 텍스트 브랜치의 $T_{cls}$를 선형 투영 $L_i$를 통해 변환하여 $I_{cls}$ 대신 주입한다.
   - $N_2 \le i < N$: 다시 기존의 시각적 $[CLS]$ 토큰 흐름을 따른다.

   이를 통해 텍스트의 전역적 카테고리 사전 정보가 시각적 인코더의 중간 단계에 주입되어, 이미지 패치들이 해당 카테고리와 관련된 영역에 더 많은 어텐션을 기울이게 된다.
3. **디코더**: 마지막 층의 출력 $E_N$은 풍부한 공간 정보를 담고 있으며, 이를 $K$-layer Transformer 기반의 경량 디코더에 입력하여 최종 세그멘테이션 결과 $O$를 얻는다.
   $$O = \text{Decoder}(E_N)$$

### ClsCLIP+ (Local Zoom-in 전략)

작은 객체의 경우 픽셀 비율이 낮아 $T_{cls}$만으로는 시각적 인코더를 충분히 가이드하기 어렵다. 이를 해결하기 위해 다음과 같은 절차를 거친다.

1. **Region Proposal**: YOLOv7을 사용하여 이미지 $I$에서 객체 후보 영역 $P_k = \{(R_i, C_i)\}$를 생성한다.
2. **Filtering**: 사용자가 원하는 세그멘테이션 대상 클래스 $A_k$와 일치하는 영역만 필터링하여 $Q_k$를 구성한다.
3. **Segmentation**: 필터링된 각 영역 $R_m \in Q_k$에 대해 ClsCLIP을 개별적으로 적용한다.
4. **Aggregation**: 서로 다른 영역에서 예측된 결과들이 겹칠 수 있으므로, 'OR 연산' 원칙을 적용하여 최종 픽셀 클래스를 결정한다.
   $$O(q) = \bigcup_{i=1}^{n} O_m(q)$$

## 📊 Results

### 실험 설정

- **데이터셋**: PASCAL-5$^i$ 및 COCO-20$^i$ 벤치마크를 사용하며, 학습 시 unseen 클래스의 이미지와 단어를 제공하지 않는 "inductive" 설정을 따른다.
- **지표**: mIoU (mean Intersection over Union)와 FB-IoU (Foreground-Background IoU)를 측정한다.
- **백본**: CLIP-ViT-B/16을 사용하였으며, 디코더는 2-layer Transformer로 구성하였다.

### 주요 결과

- **ZS3 성능 비교**: PASCAL-5$^i$ 데이터셋에서 ClsCLIP은 mIoU 56.4%를 달성하여, 기존 ZS3 방법론인 LSeg(52.3%)와 CLIPSeg(51.7%)보다 높은 성능을 보였다.
- **Few-shot 모델과의 비교**: zero-shot 설정임에도 불구하고, ClsCLIP+는 1-shot 세그멘테이션 모델인 DCAMA(69.3%)보다 높은 mIoU 71.5%를 기록하며 SOTA 성능을 달성하였다.
- **작은 객체 처리**: 정성적 결과(Figure 6)에서 ClsCLIP과 CLIPSeg가 병(bottle)이나 배(boat) 같은 작은 객체를 놓치는 반면, ClsCLIP+는 이를 매우 정교하게 분할해냄을 확인하였다.
- **Ablation Study**:
  - 텍스트 $[CLS]$ 토큰을 단순히 채널/공간 어텐션 가중치로 사용하는 것보다, 토큰 자체를 교체하는 방식이 훨씬 효과적이다.
  - 토큰 교체 위치의 경우, 너무 초기(layer 0)에 교체하면 세만틱 정렬 문제가 발생하고, 너무 늦게(마지막 3개 층) 교체하면 정보 손실이 발생한다. 얕은 층(layer 2, 3, 4)에서 교체하는 것이 최적의 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 CLIP의 텍스트 인코더가 학습한 강력한 카테고리 사전 지식을 시각적 인코더의 '입력' 단계로 끌어올림으로써, 단순한 특징 융합보다 훨씬 강력한 가이드 효과를 얻을 수 있음을 입증하였다. 특히, ViT의 셀프 어텐션 메커니즘을 통해 $T_{cls}$가 이미지 패치들과 상호작용하며 관심 영역을 스스로 찾아내게 만든 점이 핵심적인 강점이다.

다만, 몇 가지 논의할 점이 있다.
첫째, ClsCLIP+의 높은 성능이 YOLOv7의 Region Proposal 성능에 의존하고 있을 가능성이 있다. 연구진은 이를 검증하기 위해 YOLO 대신 수동 어노테이션(Manual annotation)을 사용하여 Proposal을 제공했을 때 mIoU가 23.6% 추가 상승함을 보였다. 이는 Proposal의 정밀도가 올라갈수록 ClsCLIP의 성능 또한 비례하여 상승함을 의미하며, 동시에 Proposal이 완벽하지 않더라도 ClsCLIP 자체의 분할 능력이 유효함을 시사한다.

둘째, 시각적 인코더의 가중치를 동결(freeze)한 상태에서 토큰만 교체하는 방식이 VPT(Visual Prompt Tuning)보다 성능이 좋게 나왔는데, 이는 unseen 클래스에 대해 학습 가능한 파라미터를 추가하는 것보다 이미 정렬된 CLIP의 전역 토큰을 직접 사용하는 것이 일반화 성능 측면에서 유리함을 보여준다.

## 📌 TL;DR

본 논문은 CLIP의 텍스트 $[CLS]$ 토큰을 시각적 인코더의 얕은 층에 주입하여 모델이 특정 객체 영역에 집중하게 만드는 **ClsCLIP**을 제안한다. 추가적으로 YOLOv7을 이용한 국소 영역 확대 전략을 적용한 **ClsCLIP+**를 통해 작은 객체 분할 문제를 해결하였다. 실험 결과, 본 방법론은 제로샷 설정임에도 불구하고 기존의 제로샷 및 퓨샷 세그멘테이션 모델들을 상회하는 SOTA 성능을 달성하였으며, 이는 CLIP의 전역 세만틱 정보를 인코더 단계에서 조기에 활용하는 것이 ZS3 성능 향상의 핵심임을 보여준다.

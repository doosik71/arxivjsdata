# Med-URWKV: Pure RWKV With ImageNet Pre-training For Medical Image Segmentation

Zhenhuan Zhou (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 컴퓨터 보조 진단 및 치료의 핵심 기술이다. 기존의 접근 방식은 크게 세 가지 범주인 CNN 기반, Transformer 기반, 그리고 이 둘을 결합한 하이브리드 구조로 나뉜다. 그러나 CNN은 수용 영역(Receptive Field)이 제한적이라는 단점이 있으며, Transformer는 연산 복잡도가 입력 크기에 따라 제곱으로 증가하는 Quadratic Complexity 문제로 인해 고해상도 의료 영상을 처리할 때 막대한 계산 비용과 메모리 오버헤드가 발생한다.

최근 선형 연산 복잡도를 가지면서도 강력한 장거리 모델링 능력을 갖춘 RWKV(Receptance Weighted Key Value) 모델이 대안으로 떠오르고 있다. 일부 연구가 RWKV를 의료 영상 분할에 적용하였으나, 대부분의 기존 연구는 모델을 처음부터 학습(Train from scratch)시키거나, 인코더에만 RWKV를 적용하고 디코더는 CNN을 사용하는 하이브리드 구조를 채택하고 있다. 이에 따라 대규모 데이터셋으로 사전 학습된 Vision-RWKV(VRWKV) 모델의 잠재력을 의료 영상 분할 작업에 충분히 활용하지 못하고 있다는 문제가 존재한다.

본 논문의 목표는 사전 학습된 VRWKV 인코더를 직접 활용할 수 있는 순수 RWKV 기반의 분할 아키텍처인 Med-URWKV를 제안하여, 의료 영상 분할 성능을 높이고 배포 효율성을 개선하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 분야 최초로 대규모 사전 학습된 VRWKV 인코더를 직접 재사용할 수 있는 순수 RWKV 기반의 분할 모델을 설계한 것이다.

중심적인 설계 아이디어는 U-Net 프레임워크를 기반으로 하되, 인코더부터 디코더까지 모든 구성 요소를 RWKV 구조로 통합하여 RWKV의 이점을 극대화하는 것이다. 특히 ImageNet과 ADE20K로 사전 학습된 VRWKV-Tiny 모델을 인코더로 채택함으로써 학습 수렴 속도를 가속화하고, 데이터가 부족한 의료 영상 환경에서도 높은 일반화 성능을 확보하고자 하였다.

## 📎 Related Works

기존의 의료 영상 분할 연구는 다음과 같은 흐름으로 진행되었다.
- **CNN 기반:** U-Net과 그 변형 모델들이 국소 특징 추출 능력으로 정밀한 세부 묘사에 강점을 보였으나, 전역 문맥(Global Context) 모델링 능력이 부족하였다.
- **Transformer 기반:** Self-attention 메커니즘을 통해 전역 의존성을 학습할 수 있게 되었으나, 고해상도 이미지 처리 시 연산 비용이 기하급수적으로 증가하는 문제가 발생하였다.
- **Mamba 기반:** 선형 복잡도를 통해 Transformer의 비용 문제를 해결하려 했으나, 때때로 정확도 면에서 손실이 발생하는 경향이 있다.
- **RWKV 기반:** RWKV-Unet, Zig-RiR, BSBP-RWKV, HFE-RWKV 등이 제안되었다. 그러나 이들은 주로 하이브리드 구조(CNN-RWKV)를 사용하거나 처음부터 학습시키는 방식을 취했다는 점에서 본 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
Med-URWKV는 U-Net의 대칭 구조를 따르는 순수 RWKV 기반 아키텍처이다. 전체 파이프라인은 크게 세 부분으로 구성된다.
1. **Pre-trained VRWKV Encoder:** ImageNet-based 사전 학습된 VRWKV-Tiny 모델을 사용하여 계층적 특징 $\text{X}_i \in \mathbb{R}^{\frac{H}{2^{i+1}} \times \frac{W}{2^{i+1}} \times \text{Dims}}$를 추출한다.
2. **RWKV Bottleneck Block:** 인코더의 최하단 출력 특징 $\text{X}_4$를 입력받아 추가적인 특징 추출 및 차원 축소를 수행하며, 인코더와 디코더를 연결하는 브릿지 역할을 한다.
3. **VRWKV Decoder:** 보틀넥 층의 출력을 입력받아 점진적인 패치 확장(Patch Expanding) 및 특징 디코딩을 수행한다. 이때 인코더에서 추출된 계층적 특징들이 Skip Connection을 통해 통합된다.

마지막으로 $1 \times 1$ Convolution 레이어로 구성된 Segmentation Head를 통해 최종 예측 맵 $\text{Y} \in \mathbb{N}^{H \times W \times n}$을 생성한다.

### VRWKV 블록 상세 및 방정식
VRWKV 블록은 Spatial Mix 블록과 Channel Mix 블록(FFN 역할)으로 구성된다.

**1. Spatial Mix 과정:**
입력 $\text{X}'$는 먼저 Q-shift 메커니즘을 통해 이전 타임스텝의 정보를 통합한 후, 세 개의 선형 변환을 거쳐 $R_s, K_s, V_s$ 행렬을 생성한다.
$$ R_s = \text{Q-shift}(\text{X}')\text{W}_R $$
$$ K_s = \text{Q-shift}(\text{X}')\text{W}_K $$
$$ V_s = \text{Q-shift}(\text{X}')\text{W}_V $$

**2. Bi-WKV Attention:**
이후 $\text{K}_s$와 $\text{V}_s$를 이용하여 양방향 WKV(bi-WKV) 메커니즘을 통해 어텐션 연산자 $\text{wkv}$를 계산한다.
$$ \text{Bi-WKV}(\text{K}, \text{V})_t = \frac{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} v_i + e^{u + k_t} v_t}{\sum_{i=0, i \neq t}^{T-1} e^{-(|t-i|-1)/T \cdot w + k_i} + e^{u + k_t}} $$
여기서 $w, u$는 학습 가능한 벡터이며, $t$는 현재 타임스텝을 나타낸다.

**3. 최종 출력 생성:**
$R_s$는 게이팅 메커니즘으로 작용하여 최종 출력을 제어한다.
$$ \text{O}_s = (\sigma(\text{R}_s) \odot \text{wkv})\text{W}_O $$
여기서 $\sigma$는 sigmoid 함수이며, $\odot$은 원소별 곱셈(element-wise product)이다.

### 학습 절차
- **사전 학습 모델:** ImageNet으로 학습되고 ADE20K로 미세 조정된 `supernet_vrwkv_adapter_tiny_512_160k_ade20k` 모델의 인코더 부분만 사용한다.
- **학습 전략:** 초기 5 에포크 동안은 사전 학습된 인코더의 파라미터를 동결(Frozen)하여 디코더와 보틀넥 층이 빠르게 정렬되도록 한 뒤, 이후 모든 파라미터를 해제하여 전체 모델을 학습시킨다.
- **손실 함수:** Cross-Entropy Loss와 Dice Loss를 결합하여 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋:** ISIC2017, ISIC2018(피부암), GLAS(병리), TDD(치과 X-ray), BUSI(유방 초음파), Kvasir-SEG(폴립), NKUT(사랑니) 등 총 7개의 공개 데이터셋을 사용하였다.
- **평가 지표:** Dice Similarity Coefficient (DSC)와 Intersection over Union (IoU)를 사용하였다.
- **하드웨어:** GeForce RTX 3090 24GB GPU에서 PyTorch로 구현하였다.

### 정량적 결과
Table I에 따르면, Med-URWKV는 다수의 데이터셋에서 기존 CNN 기반(UNet, ACC-Unet) 및 ViT 기반(Swin-Unet, TransUNet) 모델보다 우수하거나 경쟁력 있는 성능을 보였다.
- 특히, 처음부터 학습시킨 하이브리드 RWKV 모델인 Zig-RiR와 비교했을 때, 대부분의 지표에서 더 높은 성능을 기록하였다.
- **파라미터 효율성:** Med-URWKV의 파라미터 수는 약 $14.33\text{M}$으로, TransUNet($92.23\text{M}$)이나 UCTransNet($66.40\text{M}$)보다 훨씬 적으면서도 동등 이상의 성능을 낸다.

### 사전 학습의 효과 (Ablation Study)
BUSI 데이터셋을 대상으로 사전 학습 유무에 따른 성능을 비교한 결과, 사전 학습을 적용한 경우(w/ pre-training)가 적용하지 않은 경우(w/o pre-training)보다 DSC 기준 최종 성능이 월등히 높았다 ($\text{best: } 77.98$ vs $51.17$). 또한, 학습 초기 단계에서 수렴 속도가 훨씬 빠르고 안정적인 양상을 보였다.

## 🧠 Insights & Discussion

본 연구는 순수 RWKV 아키텍처가 의료 영상 분할 작업에서 충분한 잠재력을 가지고 있음을 입증하였다. 특히, 대규모 일반 이미지 데이터셋(ImageNet, ADE20K)으로 학습된 모델이 도메인이 다른 의료 영상 분야에서도 효과적인 특징 추출기로 작용할 수 있음을 확인하였다. 이는 하이브리드 구조를 통해 CNN의 로컬 특징 추출 능력에 의존하던 기존 RWKV 기반 모델들의 한계를 넘어, RWKV 자체의 전역 모델링 능력만으로도 충분한 성능을 낼 수 있음을 시사한다.

다만, 논문에서 언급된 한계점으로는 인코더의 규모(Scale)나 설정 변경이 분할 성능에 미치는 영향에 대한 심층적인 분석이 아직 부족하다는 점이 있다. 또한, 일반적인 VRWKV 블록 외에 의료 영상의 특성에 최적화된 전용 어텐션 메커니즘 설계가 향후 과제로 남아 있다.

## 📌 TL;DR

Med-URWKV는 사전 학습된 VRWKV 인코더를 활용한 **순수 RWKV 기반의 의료 영상 분할 모델**이다. 이 모델은 Transformer의 전역 모델링 능력과 CNN의 효율성을 동시에 잡은 선형 복잡도의 RWKV 구조를 채택하였으며, ImageNet 사전 학습을 통해 적은 파라미터($14.33\text{M}$)만으로도 기존의 무거운 ViT 기반 모델이나 하이브리드 RWKV 모델보다 뛰어난 성능과 빠른 수렴 속도를 보여주었다. 향후 의료 영상 분야에서 계산 효율적인 고성능 분할 모델의 기초가 될 가능성이 높다.
# SURGIVID: Annotation-Efficient Surgical Video Object Discovery

Caghan Koksal, Ghazal Ghazaei, and Nassir Navab (2024)

## 🧩 Problem to Solve

수술 비디오에서 도구와 해부학적 구조를 픽셀 단위로 지역화(localization)하는 것은 수술 분석, 숙련도 향상, 환자 케어 최적화 및 교육을 위한 필수적인 기초 단계이다. 그러나 이를 위한 semantic segmentation 모델을 학습시키기 위해서는 픽셀 단위의 정밀한 주석(annotation)이 필요하며, 이는 막대한 시간과 비용이 소요될 뿐만 아니라 고도의 전문 의료 지식을 갖춘 인력이 필요하다는 한계가 있다.

본 논문의 목표는 표준화된 수술 워크플로우를 통해 얻어지는 방대한 양의 수술 비디오 데이터를 활용하여, 주석에 대한 의존도를 획기적으로 낮춘 주석 효율적(annotation-efficient)인 semantic segmentation 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 자기지도 학습(self-supervised learning) 기반의 객체 발견(object discovery)과 최소한의 지도 학습을 결합하여 주석 비용을 최소화하는 것이다. 구체적인 기여 사항은 다음과 같다.

- **자기지도 객체 발견의 활용**: DINO와 MaskCut을 결합하여 수술 장면에서 가장 두드러진 도구와 해부학적 구조를 감독 없이(unsupervised) 찾아내어 pseudo-mask를 생성한다.
- **Self-training 및 정제 단계**: 생성된 pseudo-mask를 사용하여 Mask2Former 모델을 class-agnostic 설정으로 사전 학습시킨 뒤, 극소량의 주석 데이터(최대 1%까지)만으로 미세 조정(fine-tuning)하여 전체 감독 학습 모델에 근접한 성능을 달성한다.
- **약지도 학습(Weak Supervision)의 도입**: 수술 단계(surgical phase) 라벨을 약지도 학습으로 활용하여 DINO의 attention을 수술 도구에 더 집중시킴으로써 도구 지역화 성능을 약 2% 향상시킨다.

## 📎 Related Works

기존의 semantic segmentation 연구들은 주로 fully-supervised 방식에 의존해 왔으며, 최근에는 DINO, LOST, Tokencut과 같은 자기지도 기반 파운데이션 모델의 특징(feature)을 활용하려는 시도가 이어지고 있다.

- **LOST**: DINO의 마지막 attention layer에서 key feature vector를 추출하고 패치 간 유사도를 계산하여 객체를 탐지한다.
- **Tokencut**: 객체 발견 문제를 그래프 분할(graph partitioning) 문제로 정식화하여 접근한다. 그러나 Tokencut은 가장 두드러진 단일 객체만을 예측하는 경향이 있어, 여러 객체가 동시에 존재하는 장면을 처리하는 데 한계가 있다.

본 논문은 이러한 기존 접근 방식을 확장하여 의료 도메인, 특히 백내장 수술 비디오에 특화된 워크플로우를 제안하며, 특히 파운데이션 모델이 의료 데이터에서 가질 수 있는 한계를 극복하기 위해 수술 단계 라벨이라는 도메인 특화 정보를 결합한 점이 차별점이다.

## 🛠️ Methodology

### 1. Object Discovery (객체 발견)

본 프레임워크의 백본으로는 self-distillation과 Vision Transformer(ViT)를 사용하는 DINO를 채택한다. DINO의 attention map은 로컬 및 글로벌 특징을 잘 포착하므로 이를 활용해 class-agnostic pseudo segmentation mask를 생성한다.

구체적으로 MaskCut 워크플로우를 사용하며, 이미지를 동일한 크기의 패치로 나누어 DINO의 마지막 attention layer에서 key vector를 추출한다. 이후 패치 $i$와 $j$ 사이의 코사인 유사도를 통해 affinity matrix $W_{i,j}$를 계산한다.

$$W_{i,j} = \frac{K_i \cdot K_j}{\|K_i\|_2 \|K_j\|_2}$$

여기서 $K_i$와 $K_j$는 각 패치의 key feature이다. 이후 이 affinity matrix를 인접 행렬(adjacency matrix)로 사용하는 그래프 이분할(graph bi-partition) 문제로 변환하고, N-cut 문제를 해결하여 전경(foreground) binary mask를 생성한다. 이 과정은 기대하는 최대 인스턴스 수에 도달할 때까지 반복된다.

### 2. DINO Weak Supervision (약지도 학습)

수술 단계(phase) 라벨을 사용하여 DINO의 attention을 유도한다. 수술 도구의 종류와 움직임이 수술 단계를 결정하는 중요한 단서가 된다는 가설 하에, DINO의 $[CLS]$ 토큰 위에 선형 레이어를 추가하여 수술 단계 분류를 위한 multi-task learning을 수행한다. 전체 손실 함수는 다음과 같다.

$$L_{\text{multi\_task}} = \lambda_{\text{cls}}L_{\text{cls}} + \lambda_{\text{dino}}L_{\text{dino}}$$

여기서 $L_{\text{cls}}$는 Cross Entropy loss이며, $L_{\text{dino}}$는 DINO의 기본 self-supervised loss이다. $\lambda_{\text{cls}}$와 $\lambda_{\text{dino}}$는 모두 0.5로 설정되었다.

### 3. Self-Training 및 Fine-tuning

생성된 pseudo-mask를 정제하기 위해 Mask2Former를 사용한다. Mask2Former는 masked attention을 통해 transformer decoder 내에서 로컬 attention을 강제함으로써 수렴 속도와 성능을 높인다.

$$\text{X}_l = \text{softmax}(M_{l-1} + Q_l K_l^T)V_l + X_{(l-1)}$$

여기서 $M_{l-1}$은 attention mask이며, $Q_l, K_l, V_l$은 각각 query, key, value 특징이다.

학습 절차는 다음과 같다.

1. **Self-training**: MaskCut으로 생성한 pseudo-mask를 사용하여 Mask2Former를 class-agnostic 설정으로 사전 학습시킨다.
2. **Fine-tuning**: 사전 학습된 모델을 CaDIS 데이터셋의 일부(1%, 10%, 50% 등) 주석 데이터만을 사용하여 미세 조정한다. 이때 손실 함수는 mask loss와 class loss(Cross Entropy 및 Dice loss의 조합)를 사용한다.

$$L = L_{\text{mask}} + \lambda_{\text{cls}}L_{\text{cls}}, \quad L_{\text{cls}} = \lambda_{\text{ce}}L_{\text{ce}} + \lambda_{\text{dice}}L_{\text{dice}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: CaDIS 데이터셋의 Task II를 사용한다. (17개 클래스: 수술 도구 9종, 해부학적 구조 4종, 기타 4종)
- **비교 대상**: HRNetV2, OCRNet 및 Fully-supervised Mask2Former.
- **지표**: mIoU, 픽셀 단위 정확도(Pixel-wise Accuracy), $\text{mIoU}_{\text{loc}}$ (클래스 라벨을 무시하고 지역화 성능만 측정).

### 주요 결과

- **SOTA 비교**: 제안된 워크플로우는 CaDIS Task II에서 SOTA인 OCRNet보다 우수한 성능을 보였으며, Fully-supervised Mask2Former와 비교해서도 대등하거나 약간 상회하는 mIoU(80.69)를 달성하였다.
- **주석 효율성**: 단 1%의 주석 데이터만 사용한 모델이 Fully-supervised 모델과 비교하여 $\text{mIoU}_{\text{loc}}$(지역화 성능) 면에서 상당히 유사한 성능(43.33 vs 65.91, 수치적 차이는 있으나 정성적으로 유의미한 수준)을 보였다. 다만, 클래스 불균형으로 인해 일부 희귀 도구의 semantic 분류 성능(mIoU)은 낮게 나타났다.
- **약지도 학습 효과**: 수술 단계 라벨을 사용하여 DINO를 미세 조정한 경우, 도구 지역화 성능이 약 2% 향상되었다.

## 🧠 Insights & Discussion

본 연구는 극소량의 데이터만으로도 높은 수준의 객체 지역화가 가능함을 입증하였다. 특히 해부학적 구조의 경우 수술 전반에 걸쳐 형태가 일정하기 때문에 1%의 데이터만으로도 충분한 세그멘테이션이 가능했다. 반면, 수술 도구는 종류가 다양하고 등장 빈도가 다르기 때문에 더 많은 데이터나 약지도 학습(phase labels)의 도움이 필요함을 확인하였다.

**강점**:

- 주석 비용을 최대 90% 이상 절감하면서도 실용적인 수준의 성능을 유지한다.
- 도메인 지식(수술 단계)을 약지도 학습 형태로 결합하여 모델의 attention을 효과적으로 유도하였다.

**한계 및 비판적 해석**:

- 1% 데이터셋 모델에서 mIoU가 낮게 나오는 이유는 학습 데이터에 포함되지 않은 클래스가 존재하기 때문이며, 이는 단순한 지역화(localization)와 의미론적 분류(classification) 사이의 간극을 보여준다.
- 실험이 백내장 수술(CaDIS)이라는 특정 도메인에 한정되어 있어, 다른 수술 환경에서의 일반화 가능성에 대해서는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 DINO 기반의 self-supervised 객체 발견(MaskCut)과 Mask2Former의 self-training을 결합하여, 수술 비디오 세그멘테이션에 필요한 주석 양을 획기적으로 줄인 **SURGIVID** 프레임워크를 제안한다. 실험 결과, 단 1%의 주석만으로도 완전 지도 학습 모델에 근접하는 지역화 성능을 보였으며, 수술 단계 라벨을 이용한 약지도 학습을 통해 도구 탐지 능력을 추가로 향상시켰다. 이 연구는 라벨링 비용이 매우 높은 의료 영상 분야에서 방대한 미라벨링 데이터를 활용할 수 있는 실질적인 방법론을 제시했다는 점에서 의의가 있다.

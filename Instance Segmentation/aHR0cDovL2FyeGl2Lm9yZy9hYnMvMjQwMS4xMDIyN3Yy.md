# A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting

Wouter Van Gansbeke and Bert De Brabandere (2024)

## 🧩 Problem to Solve

본 논문은 Panoptic Segmentation(범용 세그멘테이션) 및 Instance Segmentation 네트워크가 가진 구조적 복잡성을 해결하고자 한다. 기존의 방식들은 인스턴스 마스크의 순열 불변성(permutation-invariance) 문제를 해결하기 위해 특수 설계된 객체 탐지(object detection) 모듈, 복잡한 손실 함수, 그리고 정교한 후처리 단계(예: Hungarian matching, NMS 등)에 크게 의존해 왔다.

이러한 접근 방식은 파이프라인을 복잡하게 만들며, 특정 작업에 종속된 설계로 인해 범용성이 떨어진다는 문제가 있다. 따라서 본 연구의 목표는 Stable Diffusion과 같은 Latent Diffusion Model(LDM)을 활용하여 이러한 복잡한 모듈 없이도 강력한 성능을 내는 단순한 생성적(generative) 세그멘테이션 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 세그멘테이션 마스크를 연속적인 잠재 공간(continuous latent space)으로 투영하고, 이미지 조건부 디퓨전 프로세스를 통해 마스크를 생성하는 것이다. 주요 기여 사항은 다음과 같다.

- **생성적 프레임워크 제안**: Stable Diffusion을 기반으로 Panoptic Segmentation을 위한 Latent Diffusion 접근 방식을 제안하였다.
- **설계의 단순화**: 기존의 객체 쿼리(object queries), 지역 제안(region proposals), 또는 Bipartite matching과 같은 복잡한 모듈을 완전히 제거하고, 이를 디퓨전 모델의 노이즈 제거(denoising) 과정으로 대체하였다.
- **마스크 인페인팅(Mask Inpainting) 구현**: 생성 모델의 특성을 이용하여, 부분적으로 주어진 마스크를 완성하는 Mask Inpainting 기능을 별도의 추가 학습 없이 수행할 수 있음을 보였다.
- **멀티태스크 확장성**: 학습 가능한 Task Embedding을 도입함으로써, 하나의 모델로 인스턴스 세그멘테이션, 시맨틱 세그멘테이션, 깊이 예측(depth prediction) 등 여러 dense prediction 작업을 동시에 수행할 수 있는 구조를 제시하였다.

## 📎 Related Works

**Panoptic Segmentation**
기존 연구들은 주로 Mask R-CNN과 같이 시맨틱 세그멘테이션과 인스턴스 세그멘테이션 브랜치를 결합하거나, Mask2Former와 같이 Object Query와 Hungarian matching을 사용하는 방식을 취했다. 그러나 이러한 방식들은 앵커 박스, NMS, 혹은 복잡한 병합 휴리스틱과 같은 특정 작업 전용 모듈에 의존하는 한계가 있다.

**General-purpose Frameworks**
Painter나 UViM과 같은 최신 연구들은 비전 작업을 생성 과정으로 정의하여 통합하려 했다. 하지만 이들은 주로 입력을 이산 토큰(discrete tokens)으로 변환하고 자기회귀(autoregressive) 모델링을 사용하는 방식을 택한다. 본 논문은 이와 달리 연속적인 잠재 공간에서의 디노이징 프로세스를 활용하여 고차원 입력의 dense prediction 작업에 더 적합한 구조를 제안한다.

**Denoising Diffusion Models**
최근 디퓨전 모델을 세그멘테이션에 적용하려는 시도가 있었으나, 인스턴스 간 구분이 불가능하거나, 여전히 객체 탐지 가중치에 의존하는 등의 한계가 있었다. 본 연구는 객체 탐지 데이터 없이 순수하게 Latent Diffusion을 통해 Panoptic Segmentation을 달성한 첫 번째 사례라고 주장한다.

## 🛠️ Methodology

LDMSeg의 전체 파이프라인은 크게 두 단계의 학습 과정으로 구성된다.

### Stage 1: 타겟 마스크 압축 (Compress Targets)
세그멘테이션 마스크는 이미지에 비해 엔트로피가 낮고 공간적 중복성이 크므로, 얕은 오토인코더(shallow autoencoder)를 통해 효율적으로 압축할 수 있다.

- **인코딩 및 디코딩**: 인스턴스 ID가 포함된 마스크 $y$를 비트 인코딩(bit-encoding)하여 입력으로 사용한다. 인코더 $f_t$는 3개의 stride 2 컨볼루션을 통해 $512 \times 512$ 크기의 마스크를 $64 \times 64$ 크기의 잠재 코드 $z_t$로 압축한다. 디코더 $g$는 전치 컨볼루션(transpose convolution)을 통해 이를 다시 복원한다.
- **손실 함수**: 재구성 오차 $L_{rec}$를 최소화하며, 구체적으로는 교차 엔트로피 손실($L_{ce}$)과 인스턴스별 마스크 손실($L_m$, BCE 및 Dice loss)을 결합하여 사용한다. 또한, 잠재 공간을 제한하기 위해 가중치 감쇠(weight decay)를 적용한다.
$$L_{AE}(w;y) = L_{ce}(w; \hat{y},y) + L_m(w; \hat{y},y) + \lambda\|w\|_2^2$$

### Stage 2: 조건부 디퓨전 모델 학습 (Train Denoising Diffusion Model)
이미지 조건 하에 노이즈가 섞인 마스크 잠재 코드를 복원하는 UNet $h_\theta$를 학습시킨다.

- **조건부 프로세스**: 이미지 $x$는 Stable Diffusion의 이미지 인코더 $f_i$를 통해 잠재 표현 $z_i$로 변환된다.
- **학습 절차**: 임의의 타임스텝 $j$에서 잠재 코드 $z_t$에 가우시안 노이즈 $\epsilon$을 더해 $\tilde{z}_j^t$를 생성한다.
$$\tilde{z}_j^t = \sqrt{\bar{\alpha}_j} z_t + \sqrt{1-\bar{\alpha}_j} \epsilon$$
- **노이즈 예측**: $\tilde{z}_j^t$와 이미지 조건 $z_i$를 채널 방향으로 결합(concatenation)하여 UNet의 입력 $z_c$로 사용하며, 실제 추가된 노이즈 $\epsilon$과 예측된 노이즈 $\hat{\epsilon}$ 사이의 평균 제곱 오차(MSE)를 최소화한다.
$$L_{LDMSeg}(\theta;\epsilon) = E_{z_c, \epsilon, j} [\|\epsilon - h_\theta(z_c, j)\|_2^2]$$

### Mask Inpainting 및 Multi-Task 확장
- **Mask Inpainting**: 추론 시, 알려진 픽셀 영역에 해당하는 잠재 코드를 고정하고 나머지 영역만 디노이징 프로세스를 통해 채우는 방식으로 수행한다.
- **Multi-Task Learning**: UNet의 cross-attention 레이어에 학습 가능한 Task Embedding을 도입하여, 쿼리하는 임베딩에 따라 시맨틱/인스턴스/깊이 맵 중 원하는 결과를 생성하게 한다.

## 📊 Results

**실험 설정**
- **데이터셋**: COCO (panoptic masks), ADE20k (semantic segmentation).
- **지표**: Panoptic Quality (PQ), mean IoU (mIoU).
- **기준선**: Mask2Former (전문가 모델), Painter, UViM (범용 모델).

**주요 결과**
- **Panoptic Segmentation**: COCO 데이터셋에서 class-agnostic 설정 시 50.8% PQ를 기록하였다. 이는 전문가 모델인 Mask2Former(59.0% PQ)보다는 낮지만, 범용 모델인 Painter(41.3% PQ)보다 우수하며 UViM(43.1% PQ)과 경쟁 가능한 수준이다.
- **Semantic Segmentation**: ADE20k 데이터셋에서 52.2% mIoU를 달성하여 UPerNet 등 기존의 여러 모델과 경쟁력 있는 성능을 보였다.
- **Mask Inpainting**: 다양한 드롭률(drop rate) 실험에서 마스크를 성공적으로 복원함을 보였다. 특히 드롭률이 70% 이하일 때 안정적인 성능을 보였으나, 매우 희소한(sparse) 입력에서는 정확도가 다소 하락하는 경향을 보였다.
- **Multi-Task**: 단일 모델이 Task Embedding 변경만으로 세 가지 작업(Instance, Semantic, Depth)을 모두 수행할 수 있음을 확인하였으며, 성능 저하가 거의 없음을 보였다.

## 🧠 Insights & Discussion

**강점 및 해석**
본 모델은 복잡한 Hungarian matching이나 객체 쿼리 없이도 디퓨전 과정의 확률적 샘플링을 통해 인스턴스 구분 문제를 해결하였다. 연구진은 디퓨전 모델의 초기 가우시안 노이즈 맵이 기존 Mask2Former나 DETR의 'Object Query'와 유사한 역할을 수행한다고 해석한다. 또한, 단순한 아키텍처 덕분에 후처리 시간이 기존 범용 모델(Painter 등) 대비 훨씬 빠르다는 이점이 있다.

**한계 및 비판적 해석**
- **추론 속도**: 디퓨전 모델 특성상 반복적인 샘플링 단계가 필요하므로, 단일 패스로 결과를 내는 판별적(discriminative) 모델보다 추론 시간이 길다.
- **해상도 손실**: 얕은 오토인코더를 통한 잠재 공간 투영 과정에서 아주 작은 배경 객체들이 소실될 가능성이 있다.
- **성능 갭**: 최신 전문가 모델(Mask2Former 등)과의 성능 격차는 여전히 존재하며, 이를 줄이기 위해 더 큰 데이터셋과 고해상도 잠재 공간의 필요성이 제기된다.

## 📌 TL;DR

본 논문은 Stable Diffusion을 응용하여 **Panoptic Segmentation을 위한 단순한 Latent Diffusion 프레임워크(LDMSeg)**를 제안한다. 핵심은 마스크를 잠재 공간으로 압축하고 이미지 조건부 디퓨전으로 이를 생성함으로써, **기존의 복잡한 객체 탐지 모듈과 매칭 알고리즘을 완전히 제거**한 것이다. 이 모델은 높은 범용성을 가져 **추가 학습 없이 마스크 인페인팅이 가능**하며, **Task Embedding을 통해 멀티태스크 학습으로 쉽게 확장**될 수 있다. 이는 향후 dense prediction 작업을 위한 통합 생성 모델의 가능성을 제시한 연구이다.
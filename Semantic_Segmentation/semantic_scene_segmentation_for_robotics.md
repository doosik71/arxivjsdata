# Semantic Scene Segmentation for Robotics

Juana Valeria Hurtado and Abhinav Valada (2024)

## 🧩 Problem to Solve

본 논문은 로봇의 자율 주행 및 상호작용을 위한 핵심 기술인 시맨틱 세그멘테이션(Semantic Segmentation)의 전반적인 기술적 흐름과 방법론을 분석한다. 로봇이 복잡하고 역동적인 실제 환경(예: 도시 지역)에서 안전하게 작동하기 위해서는 주변 환경에 대한 포괄적인 이해가 필수적이다. 특히 조명 변화, 기상 조건 등 환경적 변수가 많은 상황에서 객체의 카테고리, 위치, 모양을 픽셀 단위로 정확하게 파악하는 Dense Prediction 작업의 중요성이 매우 크다. 따라서 본 논문의 목표는 전통적인 방식부터 최신 딥러닝 기반의 시맨틱 세그멘테이션 알고리즘, 손실 함수, 데이터셋 및 평가 지표를 체계적으로 정리하여 로봇 인식 시스템의 발전을 위한 기술적 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문은 특정 새로운 모델을 제안하는 연구 논문이라기보다, 로봇 공학 관점에서의 시맨틱 세그멘테이션 기술을 집대성한 리뷰 논문(Chapter)의 성격을 띤다. 주요 기여 사항은 다음과 같다.

- **기술적 계층 구조 정립**: 객체 분류(Classification) $\rightarrow$ 객체 검출(Detection) $\rightarrow$ 객체 세그멘테이션(Object Segmentation) $\rightarrow$ 시맨틱 세그멘테이션(Semantic Segmentation)으로 이어지는 장면 이해(Scene Understanding)의 단계적 발전 과정을 명확히 정의한다.
- **심층적 아키텍처 분석**: FCN부터 Encoder-Decoder 구조, Atrous Convolution, Spatial Pyramid Pooling 등 현대적 세그멘테이션 네트워크의 핵심 설계 아이디어와 그 발전 과정을 상세히 분석한다.
- **다양한 입력 모달리티 확장**: 단일 RGB 이미지뿐만 아니라 비디오 시퀀스, 포인트 클라우드(Point Cloud), 다중 모달리티(Multimodal) 융합 방식에 대한 기술적 접근법을 제시한다.
- **실제 적용 관점의 평가**: 실시간성(Real-time) 확보를 위한 경량화 아키텍처와 로봇 공학에서 사용되는 주요 데이터셋 및 성능 평가 지표를 체계적으로 정리하여 실무적 통찰을 제공한다.

## 📎 Related Works

논문에서는 시맨틱 세그멘테이션 이전의 전통적 방법론과 초기 딥러닝 접근 방식의 한계를 설명한다.

- **전통적 방법론**: 주로 클러스터링, 외곽선(Contours), 에지(Edges) 정보 및 HOG, SURF와 같은 수동 설계된 특징(Hand-crafted features)을 사용했다. 또한 CRF(Conditional Random Fields)나 MRF(Markov Random Fields) 같은 그래픽 모델이 활용되었다. 이러한 방식은 사전 지식(A priori knowledge)에 크게 의존하며, 사람이 직접 설정한 파라미터에 따라 분할 가능한 클래스 수가 제한된다는 치명적인 한계가 있다.
- **초기 딥러닝 접근법**: VGG나 AlexNet과 같은 이미지 분류 네트워크를 가져와 Fully Connected(FC) 레이어를 미세 조정(Fine-tuning)하는 방식을 사용했다. 그러나 이 방식은 과적합(Overfitting) 문제가 심각하고 학습 시간이 매우 길며, 픽셀 단위의 정밀한 특징을 추출하기에는 판별력이 부족했다.

## 🛠️ Methodology

본 논문에서 다루는 시맨틱 세그멘테이션의 핵심 방법론은 다음과 같다.

### 1. 네트워크 아키텍처의 진화

- **Fully Convolutional Networks (FCN)**: FC 레이어를 모두 컨볼루션 레이어로 대체하여 입력 이미지의 크기에 상관없이 Dense Prediction을 가능하게 한 구조이다.
- **Encoder-Decoder 구조**: 인코더는 다운샘플링을 통해 깊은 문맥 정보(Contextual information)를 캡처하고, 디코더는 업샘플링(Deconvolution 등)을 통해 공간 정보를 복원하여 원래 해상도로 되돌린다. (예: U-Net의 Skip Connection은 인코더의 세밀한 정보를 디코더에 직접 전달하여 경계 복원력을 높인다.)
- **Context Exploitation 기술**:
  - **Dilated (Atrous) Convolution**: 필터 사이의 간격을 띄워 파라미터 수 증가 없이 수용 영역(Receptive Field)을 넓히는 기법이다.
  - **Spatial Pyramid Pooling (SPP/ASPP)**: 다양한 비율의 샘플링을 통해 서로 다른 스케일의 객체 정보를 동시에 캡처한다.
  - **Image Pyramid**: 동일한 모델에 다양한 크기의 입력을 넣어 광역 문맥과 세부 디테일을 동시에 확보한다.

### 2. 손실 함수 (Loss Functions)

- **Pixel-wise Cross Entropy Loss**: 각 픽셀의 예측 클래스와 정답을 비교하여 평균을 내는 방식이다.
$$L_{CE}(p, y) = -\sum_c y_{x,c} \log p_{x,c}$$
여기서 $c$는 클래스 라벨, $y_{x,c}$는 정답 여부(0 또는 1), $p$는 예측 확률이다. 클래스 불균형 문제를 해결하기 위해 가중치를 부여하기도 한다.
- **Dice Loss**: 예측 영역과 정답 영역의 겹침 정도(Overlap)를 측정하는 Dice Coefficient를 기반으로 한다.
$$\text{Dice}(y, c) = \frac{2|A \cap B|}{|A| + |B|}$$
$$L_D(y, c) = 1 - \text{Dice}(y, c)$$
이는 클래스 불균형이 심한 데이터셋에서 특히 유용하다.

### 3. 입력 데이터 확장 및 융합

- **Point Cloud Segmentation**: 3D 데이터를 직접 처리하는 Point-based, 복셀화하는 Voxel-based, 2D로 투영하는 Projection-based 방법으로 나뉜다.
- **Multimodal Fusion**: RGB-D 등 서로 다른 센서 데이터를 융합하는 방식은 세 가지로 구분된다.
  - **Early Fusion**: 입력 단계에서 채널을 쌓아(Channel stacking) 하나의 네트워크에 입력한다.
  - **Late Fusion**: 각 모달리티별 독립된 네트워크를 거친 후 마지막 단계에서 특징 맵을 결합한다.
  - **Hybrid Fusion**: 위 두 방식의 장점을 결합하여 여러 단계에서 특징을 융합한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 업계에서 통용되는 벤치마크와 지표를 체계적으로 정리하여 제시한다.

### 1. 주요 데이터셋

- **Outdoor**: Cityscapes (유럽 도시 거리), KITTI (자율주행 종합), BDD100K (다양한 날씨/시간대), IDD (비정형 인도 환경) 등.
- **Indoor**: NYU-Depth V2, ScanNet 등 RGB-D 기반 데이터셋.
- **General**: PASCAL VOC, MS COCO, ADE20K 등 범용 객체 데이터셋.

### 2. 평가 지표

- **Pixel Accuracy (PA)**: 전체 픽셀 중 맞게 분류된 픽셀의 비율이다.
- **Intersection over Union (IoU)**: 예측 영역과 정답 영역의 교집합을 합집합으로 나눈 값이다.
$$\text{IoU} = \frac{TP}{TP + FP + FN}$$
mIoU(mean IoU)는 모든 클래스에 대한 IoU의 평균이며, PA보다 클래스 불균형에 강건한 지표로 평가된다.
- **Computational Complexity**: 로봇 탑재를 위해 Runtime, Memory Usage, 그리고 FLOPs(Floating Point Operations Per Second)를 측정한다.

## 🧠 Insights & Discussion

### 강점 및 가치

본 논문은 시맨틱 세그멘테이션의 이론적 배경부터 실무적인 구현 고려사항(실시간성, 메모리 제한)까지 매우 폭넓게 다루고 있다. 특히 단순한 알고리즘 나열이 아니라, 왜 Atrous Convolution이 필요한지, 왜 Encoder-Decoder 구조가 도입되었는지에 대한 논리적 흐름을 제공한다.

### 한계 및 비판적 해석

- **데이터 의존성**: 논문에서도 언급되었듯이, 지도 학습(Supervised Learning) 기반 모델은 방대한 양의 픽셀 단위 라벨링 데이터가 필요하며, 이는 비용 면에서 매우 비효율적이다.
- **실시간성 vs 정확도**: BiSeNet이나 ESPNet 같은 경량 모델이 제안되었으나, 여전히 높은 정확도를 유지하면서 초저지연(Ultra-low latency)을 달성하는 것은 로봇 하드웨어 제약상 어려운 과제이다.
- **정적 이미지 중심**: 비디오 세그멘테이션에 대한 언급이 있으나, 여전히 많은 분석이 단일 프레임 기반에 치중되어 있어, 시간적 일관성(Temporal coherence)을 완전히 해결한 최신 기법에 대한 심층 분석은 부족한 편이다.

## 📌 TL;DR

본 보고서는 로봇 자율 주행을 위한 시맨틱 세그멘테이션 기술의 전 과정을 분석한 리뷰이다. FCN에서 시작해 U-Net, DeepLab, PSPNet으로 이어지는 아키텍처의 발전과 Cross Entropy/Dice Loss 같은 학습 목표, 그리고 Cityscapes/KITTI와 같은 표준 벤치마크를 상세히 다룬다. 특히 단일 센서를 넘어 다중 모달리티 융합과 포인트 클라우드 처리까지 확장하여 설명한다. 향후 연구는 라벨링 비용을 줄이기 위한 자기지도 학습(Self-supervised learning)과 실시간 추론 성능 최적화, 그리고 시맨틱과 인스턴스 세그멘테이션을 통합한 Panoptic Segmentation 방향으로 나아갈 것으로 전망된다.

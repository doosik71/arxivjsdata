# FedFMS: Exploring Federated Foundation Models for Medical Image Segmentation

Yuxi Liu, Guibo Luo, and Yuesheng Zhu (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단에서 매우 중요한 역할을 수행한다. 최근 Segment Anything Model (SAM)과 같은 강력한 Foundation Model이 등장하며 시각적 분할 성능이 비약적으로 향상되었으며, 이를 의료 영상 분야에 적용하려는 시도가 이어지고 있다. 그러나 의료 영상 데이터는 개인정보 보호 및 보안 문제가 매우 민감하여, 데이터를 한곳에 모아 학습시키는 중앙 집중식 저장 및 공유 방식(Centralized storage and sharing)을 적용하기 어렵다.

또한, 대규모 모델을 학습시키기 위해서는 방대한 양의 데이터를 전송해야 하므로 통신 비용이 증가하고 전송 지연이 발생하는 문제가 있다. 기존의 Federated Learning (연합 학습) 프레임워크 내에서 의료 영상 분할을 위한 Foundation Model을 배포하려는 시도는 매우 드물었으며, 특히 Non-Independent and Identically Distributed (Non-IID) 데이터셋 환경에서 Foundation Model이 중앙 집중식 학습과 유사한 성능을 유지할 수 있는지, 그리고 대규모 모델의 통신 및 학습 효율성을 어떻게 개선할 수 있는지는 여전히 미개척 영역으로 남아 있다.

본 논문의 목표는 연합 학습 환경에서 SAM을 의료 영상 분할에 적용한 FedFMS(Federated Foundation models for Medical image Segmentation) 프레임워크를 제안하고, 이를 통해 데이터 프라이버시를 유지하면서도 중앙 집중식 학습에 근접한 성능을 달성하고 학습 효율성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 연합 학습 환경에서 Foundation Model의 가능성을 탐색하고, 효율적인 학습 구조를 설계한 것에 있다. 중심적인 설계 아이디어는 다음과 같다.

1.  **FedSAM 및 FedMSA 개발**: 모든 파라미터를 미세 조정하는 FedSAM과, 어댑터(Adapter)와 디코더(Decoder)만을 효율적으로 학습시키는 FedMSA라는 두 가지 연합 학습 프레임워크를 제안하였다.
2.  **효율적인 파라미터 업데이트**: FedMSA에서는 SAM의 이미지 인코더(Image Encoder)를 동결(Frozen)시키고, 경량화된 Adapter 모듈만을 학습함으로써 통신 비용과 계산 비용을 획기적으로 줄였다.
3.  **멀티 센터 벤치마크 데이터셋 구축**: 다양한 모달리티와 기관에서 수집된 Non-IID 의료 영상 데이터셋(전립선암, 뇌종양, 세포핵, 안저 영상)을 구축하여 연합 학습 모델의 성능을 종합적으로 평가하였다.
4.  **중앙 집중식 학습과의 성능 비교**: 연합 학습 기반의 Foundation Model이 중앙 집중식 학습 모델과 비교하여 성능 차이가 거의 없음을 입증하며, 의료 분야에서의 실용 가능성을 제시하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들을 바탕으로 하며, 그 한계를 극복하고자 한다.

-   **SAM (Segment Anything Model)**: 방대한 데이터(SA-1B)로 사전 학습된 강력한 모델이지만, 일반적인 자연 이미지 기반으로 학습되었기에 의료 영상과 같은 특수 도메인에 그대로 적용하기에는 한계가 있다. 이를 보완하기 위해 Medical SAM Adapter (MSA) 등 다양한 미세 조정(Fine-tuning) 기법이 제안되었다.
-   **Federated Learning (FL)**: 데이터를 로컬에 유지한 채 모델 파라미터만을 공유하여 프라이버시를 보호하는 학습 방식이다. 하지만 대규모 모델(Foundation Model)을 FL에 적용할 경우, 모델의 크기로 인해 발생하는 막대한 통신 오버헤드가 주요 병목 구간이 된다.
-   **기존 의료 영상 분할 모델 (U-Net, nnU-Net)**: 전통적인 CNN 기반 모델들은 의료 영상 분할에서 효과적이지만, 데이터가 부족한 도메인이나 Non-IID 환경에서는 일반화 성능이 떨어지는 경향이 있다.

본 논문은 이러한 배경 하에 Foundation Model의 강력한 사전 지식(Prior knowledge)을 연합 학습 프레임워크에 통합함으로써, Non-IID 데이터셋에서도 안정적인 성능을 내면서도 효율적인 통신 구조를 가지는 모델을 구현하였다.

## 🛠️ Methodology

### 전체 시스템 구조
FedFMS는 여러 개의 로컬 클라이언트(Client)와 하나의 글로벌 서버(Server)로 구성된다. 각 클라이언트는 자신의 로컬 데이터를 보유하며, 서버는 클라이언트들로부터 수집한 파라미터를 집계(Aggregation)하여 전역 모델(Global Model)을 업데이트한다.

### 주요 구성 요소 및 모델 설계
본 연구에서는 SAM의 ViT-B/16 및 ViT-L/16 변형 모델을 사용하였다. 의료 영상의 다중 클래스 분할(Multi-class segmentation)을 위해, 기존 SAM의 프롬프트 입력 및 프롬프트 인코더를 제거하고, 디코더 뒤에 $1 \times 1$ 컨볼루션 커널을 가진 **Multi-class Segmentation Header**를 추가하여 출력 마스크를 $H \times W \times c$ (여기서 $c$는 클래스 수) 형태로 매핑하였다.

1.  **FedSAM**: 
    -   사전 학습된 SAM의 모든 파라미터를 각 클라이언트에서 미세 조정한다.
    -   로컬 업데이트 대상: 이미지 인코더 파라미터 $W^{(e)}$ 및 디코더 파라미터 $W^{(d)}$.
2.  **FedMSA**:
    -   MSA(Medical SAM Adapter) 구조를 기반으로 하며, 인코더 내의 사전 학습된 파라미터는 동결시킨다.
    -   대신 각 ViT 블록에 Down-projection $\rightarrow$ ReLU $\rightarrow$ Up-projection으로 구성된 경량 **Adapter** 모듈을 삽입한다.
    -   로컬 업데이트 대상: 어댑터 파라미터 $W^{(a)}$ 및 디코더 파라미터 $W^{(d)}$.

### 학습 및 집계 절차
-   **손실 함수**: 모든 클라이언트는 Binary Cross Entropy (BCE) Loss를 사용하여 학습을 진행한다.
-   **파라미터 집계**: 서버는 **FedAvg** 알고리즘을 사용하여 각 클라이언트의 데이터를 가중치로 하여 파라미터를 평균낸다. 집계 식은 다음과 같다.
$$W_{t+1} \leftarrow \frac{1}{\sum_{k=1}^{K} N_{k}^{(local)}} \sum_{k=1}^{K} (N_{k}^{(local)} \cdot W_{t+1,k})$$
여기서 $N_{k}^{(local)}$은 $k$번째 클라이언트의 데이터 양이며, $W_{t+1,k}$는 해당 클라이언트의 업데이트된 파라미터이다.

## 📊 Results

### 실험 설정
-   **데이터셋**: Prostate Cancer, Brain Tumor, Nuclei, Fundus 등 4가지 Non-IID 데이터셋을 사용하였다.
-   **평가 지표**: Dice coefficient와 IoU (Intersection over Union)를 사용하였다.
-   **비교 대상**: 중앙 집중식 학습 모델(SAM, MSA), 기존 연합 학습 모델(FedU-Net, FednnU-Net).
-   **환경**: PyTorch 기반, NVIDIA A800 GPU, 총 100 라운드의 연합 학습 진행.

### 주요 결과
1.  **분할 성능**: 
    -   FedSAM과 FedMSA 모두 중앙 집중식 학습 모델(SAM, MSA)과 비교했을 때 통계적으로 유의미한 차이가 없는 유사한 성능을 보였다 ($p > 0.5$).
    -   전통적인 모델인 FedU-Net 및 FednnU-Net보다 월등히 높은 성능을 기록하였다. 특히 FednnU-Net은 Non-IID 데이터셋에서 학습이 불안정했으나, FedFMS는 Foundation Model의 풍부한 배경 지식 덕분에 높은 강건성(Robustness)을 보였다.
2.  **모델 효율성 (FedMSA vs FedSAM)**:
    -   **학습 파라미터 수**: FedMSA (14.7 M) $\ll$ FedSAM (93.7 M).
    -   **계산 비용 (FLOPs)**: FedMSA (5.7 T) $\ll$ FedSAM (13.4 T).
    -   **학습 시간 및 메모리**: FedMSA가 학습 시간과 GPU 메모리 사용량 면에서 훨씬 효율적이었다. 다만, 추론 시간(Predicting time)은 Adapter 모듈의 추가로 인해 FedMSA가 아주 약간 더 소요되었다.
3.  **사전 학습(Pre-training)의 영향**:
    -   사전 학습된 파라미터를 제거한 변형 모델($-PT$)의 경우 성능이 급격히 저하되었다.
    -   일부 사례에서는 FedSAM(-PT)나 FedMSA(-PT)가 가벼운 모델인 FedU-Net보다 성능이 낮게 나타났다. 이는 Foundation Model의 효과가 연합 학습 환경에서도 사전 학습된 지식에서 기인함을 시사한다.

## 🧠 Insights & Discussion

본 논문은 Foundation Model을 연합 학습 프레임워크에 성공적으로 통합함으로써 다음과 같은 통찰을 제공한다.

첫째, **Foundation Model의 강건성**이다. Non-IID 데이터셋에서는 클라이언트 간 데이터 분포 차이로 인해 전역 모델의 수렴 방향이 일치하지 않는 문제가 발생하는데, SAM과 같은 대규모 모델은 이미 방대한 데이터를 통해 일반화된 특징을 학습했으므로 이러한 불일치 문제를 완화하고 안정적인 수렴을 가능하게 한다.

둘째, **효율성과 성능의 트레이드-오프**이다. 모든 파라미터를 학습시키는 FedSAM이 일부 데이터셋에서 더 높은 성능을 보였으나, FedMSA는 극히 적은 파라미터 업데이트만으로도 이에 근접한 성능을 냈다. 이는 실무적인 의료 현장에서 통신 대역폭이 제한적일 때 Adapter 기반의 연합 학습이 매우 유용한 전략이 될 것임을 의미한다.

셋째, **사전 학습 지식의 필수성**이다. 실험 결과, Foundation Model의 구조 자체보다 그 안에 담긴 '사전 학습된 가중치'가 성능의 핵심이다. 이는 의료 영상과 같이 데이터 수집이 어려운 도메인에서 일반 도메인의 대규모 모델을 가져와 연합 학습으로 미세 조정하는 전략이 매우 유효함을 입증한다.

## 📌 TL;DR

본 연구는 의료 영상 분할을 위해 SAM을 연합 학습 프레임워크에 적용한 **FedFMS**를 제안하였다. 전체 파라미터를 학습하는 **FedSAM**과 어댑터만을 학습하는 효율적인 **FedMSA** 두 가지 버전을 구현하였으며, 실험을 통해 데이터 프라이버시를 유지하면서도 중앙 집중식 학습과 대등한 성능을 낼 수 있음을 확인하였다. 특히 FedMSA는 통신 및 계산 비용을 획기적으로 줄여 실용성을 높였다. 이 연구는 향후 의료 분야에서 프라이버시 보호형 Foundation Model 배포의 중요한 기준점이 될 것으로 기대된다.
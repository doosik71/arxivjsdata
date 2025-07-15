# LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search
Bin Yan, Houwen Peng, Kan Wu, Dong Wang, Jianlong Fu, and Huchuan Lu

## 🧩 Problem to Solve
최근 몇 년간 객체 추적 기술은 상당한 발전을 이루었지만, 최신 추적 모델들은 점점 더 무겁고 연산 비용이 높아져 자원 제약이 있는 환경(예: 드론, 산업용 로봇, 운전 보조 시스템)에 배포하기 어렵다는 문제가 있습니다. 기존의 모델 압축 기법은 성능 저하를 초래하며, 새로운 경량 모델을 수작업으로 설계하는 것은 많은 엔지니어링 비용과 전문 지식을 필요로 합니다. 이 논문은 객체 추적을 위한 가볍고 효율적인 신경망 아키텍처를 자동으로 탐색하는 방법을 제시하여 이러한 배포의 어려움을 해결하고자 합니다.

## ✨ Key Contributions
*   **객체 추적을 위한 신경망 아키텍처 자동 설계의 첫 시도**: 객체 추적에 특화된 새로운 원샷(One-Shot) NAS(Neural Architecture Search) 방식을 제안하여 유망한 아키텍처를 탐색합니다.
*   **경량화된 탐색 공간 및 전용 탐색 파이프라인 설계**: 효율적인 추적 아키텍처 구성을 위한 경량 블록(예: Depthwise Separable Convolutions, Inverted Residual Structure)으로 구성된 탐색 공간을 설계하고, 이를 효과적으로 탐색하는 전용 파이프라인을 구축했습니다.
*   **뛰어난 성능과 효율성**: 제안된 LightTrack이 탐색한 추적기는 기존 수작업 SOTA(State-Of-The-Art) 모델보다 훨씬 적은 FLOPs와 파라미터를 사용하면서도 동등하거나 더 우수한 성능을 달성합니다.
*   **다양한 자원 제약 플랫폼에서의 실시간 배포 가능**: 탐색된 추적기는 Snapdragon, Apple A 시리즈, Kirin 등 모바일 칩셋에서 기존 모델보다 훨씬 빠르게 실행되어 실시간 추적을 가능하게 합니다.

## 📎 Related Works
*   **객체 추적 (Object Tracking)**: Siamese 네트워크 기반의 추적기(SiamFC, SINT, SiamRPN, SiamRPN++, Ocean, ATOM, DiMP 등)들이 높은 성능을 달성했지만, 그 대가로 모델 복잡성과 계산 비용이 크게 증가했습니다. 특히 SiamRPN++(48.9G FLOPs)나 Ocean(20.3G FLOPs)은 모바일 환경(보통 < 600M FLOPs)에 비해 훨씬 무겁습니다.
*   **신경망 아키텍처 탐색 (Neural Architecture Search, NAS)**: 강화 학습 또는 진화 알고리즘 기반의 초기 NAS 방법들은 탐색 비용이 매우 높았습니다. 최근에는 가중치 공유(weight sharing)를 통한 원샷 NAS(One-Shot NAS) 방식이 많이 연구되어 탐색 비용을 크게 줄였습니다. 미분 가능한(differentiable) NAS도 인기를 얻고 있습니다. NAS는 주로 이미지 분류에 적용되었으며, 최근에는 이미지 분할(Image Segmentation)이나 객체 탐지(Object Detection)로 확장되었습니다.
*   **LightTrack의 차별점**: NAS를 객체 추적에 적용한 최초의 연구입니다. 또한, 기존 DetNAS가 백본 네트워크만 탐색하고 헤드 네트워크를 고정했던 것과 달리, LightTrack은 백본과 헤드 아키텍처를 동시에 탐색하여 전체 추적 작업에 최적화된 구조를 찾습니다. 아울러, 경량 아키텍처 탐색에 특화된 새로운 탐색 공간을 설계했습니다.

## 🛠️ Methodology
LightTrack은 객체 추적을 위한 원샷 NAS 기법으로, 세 가지 주요 단계로 구성됩니다:

1.  **백본 슈퍼넷(Backbone Supernet) 사전 학습**:
    *   모든 가능한 백본 아키텍처를 인코딩하는 하나의 백본 슈퍼넷 $N_{b}$를 구축합니다.
    *   이 슈퍼넷은 ImageNet 데이터셋을 사용하여 이미지 분류 손실($L_{\text{cls\_pre-train}}$)로 한 번만 사전 학습됩니다.
    *   사전 학습 시, 각 배치에서 랜덤하게 하나의 경로(서브넷)만 샘플링하여 가중치를 업데이트하는 단일 경로 균일 샘플링(single-path uniform sampling) 방식을 사용합니다.
    *   이렇게 학습된 가중치 $W^{p}_{b}$는 모든 백본 서브넷에 공유되어 초기화에 사용됩니다.

2.  **추적 슈퍼넷(Tracking Supernet) 학습**:
    *   백본 슈퍼넷 $N_{b}$와 헤드 슈퍼넷 $N_{h}$로 구성된 전체 추적 슈퍼넷 $N = \{N_{b}, N_{h}\}$을 구축합니다.
    *   백본 슈퍼넷은 ImageNet으로 사전 학습된 $W^{p}_{b}$로 초기화되고, 헤드 슈퍼넷은 랜덤 초기화됩니다.
    *   이 슈퍼넷은 Youtube-BB, ImageNet VID/DET, COCO, GOT-10K 등 추적 데이터셋을 사용하여 추적 손실($L_{\text{trk\_train}}$)로 공동으로 학습됩니다.
    *   추적 손실은 전경-배경 분류를 위한 이진 교차 엔트로피 손실과 바운딩 박스 회귀를 위한 IoU 손실을 포함합니다.
    *   이 단계에서도 단일 경로 균일 샘플링 방식을 사용하여 백본과 헤드 슈퍼넷에서 하나의 랜덤 경로를 샘플링하여 학습합니다.

3.  **진화 알고리즘(Evolutionary Algorithm)을 이용한 아키텍처 탐색**:
    *   학습된 추적 슈퍼넷 위에서 진화 알고리즘을 사용하여 최적의 아키텍처($\alpha^{*}_{b}$, $\alpha^{*}_{h}$)를 탐색합니다.
    *   탐색 목표는 추적 정확도($Acc^{\text{trk}}_{\text{val}}$)를 최대화하면서, FLOPs($Flops(\alpha^{*}_{b}) + Flops(\alpha^{*}_{h}) \leq Flops_{\text{max}}$)와 파라미터 수($Params(\alpha^{*}_{b}) + Params(\alpha^{*}_{h}) \leq Params_{\text{max}}$)와 같은 제약 조건을 만족하는 것입니다.
    *   진화 알고리즘은 초기 아키텍처 집단에서 시작하여, 상위 k개의 아키텍처를 부모로 선택한 후 돌연변이(mutation) 및 교배(crossover)를 통해 자식 네트워크를 생성하며 새로운 세대를 만듭니다.
    *   평가 시에는 배치 정규화(Batch Normalization) 통계를 각 서브넷에 대해 재계산하여 정확한 성능 추정을 보장합니다.

**탐색 공간 (Search Space)**:
*   **경량 빌딩 블록**: Depthwise Separable Convolutions (DSConv) 및 Squeeze-Excitation 모듈이 적용된 Mobile Inverted Bottleneck (MBConv)을 사용하여 효율적인 아키텍처를 구성합니다.
*   **백본 공간 ($A_{b}$)**: 6가지 기본 블록(MBConv with kernel sizes of $\{3,5,7\}$ and expansion rates of $\{4,6\}$)으로 구성되며, 4개의 스테이지와 총 16의 스트라이드를 가집니다. 약 $7.8 \times 10^{10}$개의 백본 아키텍처를 포함합니다.
*   **헤드 공간 ($A_{h}$)**: 분류(classification)와 회귀(regression) 두 개의 브랜치로 구성되며, 각각 최대 8개의 탐색 가능한 레이어를 포함합니다. DSConv 블록은 커널 사이즈 $\{3,5\}$, 채널 수 $\{128,192,256\}$를 선택할 수 있으며, 스킵 연결(skip connection)도 포함됩니다. 약 $3.9 \times 10^{8}$개의 헤드 아키텍처를 포함합니다.
*   **출력 피처 레이어 탐색**: 백본 슈퍼넷의 마지막 8개 블록 중 임의의 레이어를 출력 피처 레이어로 선택하여, 최적의 피처 추출 레이어를 자동으로 결정할 수 있도록 합니다.
*   정의된 탐색 공간은 기존 수작업 모델보다 훨씬 가벼운 208M~1.4G FLOPs, 0.2M~5.4M 파라미터 범위의 아키텍처를 포함합니다.

## 📊 Results
LightTrack은 모델 성능, 복잡성 및 런타임 속도 측면에서 기존 수작업 설계 객체 추적기와 비교하여 뛰어난 결과를 보여주었습니다. LightTrack은 LightTrack-Mobile (≤600M Flops, ≤2M Params), LargeA (≤800M Flops, ≤3M Params), LargeB (≤800M Flops, ≤4M Params) 세 가지 버전으로 제공됩니다.

*   **VOT-19**: LightTrack-Mobile은 기존 SOTA 오프라인 추적기(SiamRPN++, SiamFC++)보다 10배 이상 적은 FLOPs와 파라미터를 사용하면서도 우수한 EAO 성능(예: LightTrack-Mobile은 0.333 EAO, SiamRPN++는 0.292 EAO)을 달성했습니다. 온라인 업데이트를 사용하는 ATOM, DiMP$_{r}$과 비교해도 경쟁력 있는 성능을 보였습니다.
*   **GOT-10K**: LightTrack은 SOTA 성능을 달성했으며, LightTrack-Mobile은 SiamFC++(G) 및 Ocean(off)보다 우수한 AO 점수를 기록했습니다. LightTrack-LargeB는 DiMP-50보다 1.2% 높은 AO 점수(0.623 vs 0.611)를 달성하면서도 파라미터 수는 8배 적었습니다 (3.1M vs 26.1M).
*   **TrackingNet**: LightTrack-Mobile은 DiMP-50보다 0.8% 높은 정밀도(P)를 기록했으며, SiamRPN++ 및 DiMP-50과 유사한 P$_{norm}$ 및 AUC를 달성하면서도 파라미터 수는 각각 96%, 92% 더 적었습니다.
*   **LaSOT**: LightTrack-LargeB는 0.555의 성공 점수를 달성하여 SiamFC++(G) 및 Ocean-offline을 능가했습니다. 온라인 추적기인 DiMP-18보다도 성공 점수가 2.1% 높았고, 파라미터 수는 12배 적었습니다 (3.1M vs 39.3M).
*   **속도 (Run-time Speed)**: Snapdragon 845 Adreno 630 GPU, Apple A10 Fusion GPU, Kirin 985 Mali-G77 GPU와 같은 모바일 플랫폼에서 SiamRPN++나 Ocean이 실시간 속도(25fps 미만)를 달성하지 못하는 반면, LightTrack은 훨씬 더 효율적으로 작동했습니다. 예를 들어, Snapdragon 845 Adreno 630 GPU에서 LightTrack은 Ocean보다 12배 빠르게(38.4fps vs 3.2fps) 실행되었습니다. 이는 LightTrack이 자원 제약이 있는 환경에서 실시간 배포가 가능함을 입증합니다.

## 🧠 Insights & Discussion
*   **컴포넌트별 분석**: 백본, 출력 레이어, 헤드 각 컴포넌트의 자동 탐색이 전반적인 추적 성능 향상에 기여함을 입증했습니다. 특히, 수작업 MobileNetV3-large 백본이 객체 추적에 최적화되어 있지 않음을 보여주며, NAS를 통한 백본 탐색이 2.4% EAO 성능 향상을 가져왔습니다. 출력 피처 레이어를 탐색하는 것이 0.307까지 성능을 더욱 향상시켰고, 탐색 가능한 헤드 아키텍처 또한 수작업 헤드보다 2.9% EAO 이득을 제공했습니다.
*   **ImageNet 사전 학습의 영향**: ImageNet 사전 학습이 최종 추적 성능에 매우 중요함을 확인했습니다. 사전 학습이 없는 경우 성능이 크게 저하되며, 사전 학습 에폭 수가 늘어날수록 추적 정확도가 향상됩니다.
*   **탐색된 아키텍처 분석**:
    *   백본 블록의 약 50%가 7x7 커널 사이즈를 사용하는 MBConv를 채택했습니다. 이는 넓은 수용 필드(receptive field)가 객체 위치 정확도를 향상시킬 수 있음을 시사합니다.
    *   탐색된 아키텍처는 두 번째 마지막 블록을 피처 출력 레이어로 선택했습니다. 이는 추적 네트워크가 반드시 고수준 피처만을 선호하지 않을 수 있음을 나타냅니다.
    *   분류 브랜치가 회귀 브랜치보다 더 적은 레이어를 포함했습니다. 이는 대략적인 객체 위치 파악이 정밀한 바운딩 박스 회귀보다 상대적으로 쉽다는 사실에 기인할 수 있습니다.
*   **의미**: 이러한 발견들은 미래 추적 네트워크 설계에 중요한 통찰력을 제공할 수 있습니다. LightTrack은 학술적 모델과 산업적 배포 간의 격차를 줄일 수 있는 가능성을 제시합니다.

## 📌 TL;DR
**문제**: 최신 객체 추적 모델은 무거워 자원 제약이 있는 모바일/엣지 장치에 배포하기 어렵다.
**방법**: LightTrack은 객체 추적을 위한 원샷 NAS 프레임워크를 제안한다. ImageNet 사전 학습된 백본 슈퍼넷과 헤드 슈퍼넷을 추적 데이터로 공동 학습한 후, 진화 알고리즘으로 경량 빌딩 블록 기반의 효율적인 아키텍처를 탐색한다.
**핵심 결과**: LightTrack이 탐색한 모델은 기존 SOTA 모델보다 FLOPs와 파라미터가 10배 이상 적으면서도 동등하거나 더 우수한 추적 성능을 달성하며, 모바일 칩셋에서 기존 모델 대비 5~17배 빠르게 실시간으로 실행 가능하다.
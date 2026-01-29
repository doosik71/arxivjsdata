# PolarMask: Single Shot Instance Segmentation with Polar Representation
Enze Xie, Peize Sun, Xiaoge Song, Wenhai Wang, Ding Liang, Chunhua Shen, Ping Luo

## 🧩 Problem to Solve
인스턴스 분할(instance segmentation)은 이미지 내 각 객체의 위치와 픽셀 단위의 마스크를 모두 예측해야 하므로 매우 어려운 과제입니다. 기존의 투 스테이지(two-stage) 방법(예: Mask R-CNN)은 높은 정확도를 보이지만 계산 비용이 많이 들고 속도가 느립니다. 원 스테이지(one-stage) 방법은 더 간단하고 효율적이지만, 픽셀 단위 마스크 예측의 복잡성으로 인해 단일 샷(single-shot) 방식에서는 성능 저하가 발생할 수 있습니다. 이 논문은 기존 객체 감지기(object detector)에 쉽게 통합될 수 있는 개념적으로 간단하고, 계산 효율적이며, 유연한 단일 샷 인스턴스 분할 프레임워크를 제안하고자 합니다.

## ✨ Key Contributions
*   **새로운 프레임워크 제안**: 인스턴스 마스크를 폴라 좌표계(polar coordinate)로 모델링하는 새로운 단일 샷, 앵커 박스 프리(anchor-box free) 인스턴스 분할 프레임워크인 PolarMask를 제안합니다. 이는 인스턴스 분할 문제를 인스턴스 중심 분류(instance center classification)와 밀집 거리 회귀(dense distance regression)의 두 가지 병렬 작업으로 전환합니다.
*   **Polar IoU Loss 도입**: 밀집 거리 회귀의 최적화를 용이하게 하고 정확도를 크게 향상시키는 Polar IoU Loss를 제안합니다. 이는 표준 smooth-$l_1$ loss보다 우수한 성능을 보이며, 회귀 손실과 분류 손실 간의 균형을 자동으로 유지하는 이점이 있습니다.
*   **Polar Centerness 제안**: FCOS의 "Centerness" 개념을 인스턴스 분할에 맞게 개선한 Polar Centerness를 도입하여, 특히 AP$_{75}$와 같은 엄격한 지역화 지표에서 성능을 크게 향상시킵니다. 이는 고품질 마스크를 생성하는 중심점 샘플에 더 높은 가중치를 부여합니다.
*   **단순하고 유연한 인스턴스 분할 프레임워크**: 인스턴스 분할의 설계 및 계산 복잡성이 바운딩 박스 객체 감지와 동일하게 단순화될 수 있음을 최초로 보여줍니다. 제안된 프레임워크는 더 복잡한 기존 원 스테이지 방법들과 비교하여 경쟁력 있는 정확도를 달성합니다.

## 📎 Related Works
*   **투 스테이지 인스턴스 분할**: Mask R-CNN, FCIS, PANet, Mask Scoring R-CNN 등은 바운딩 박스를 먼저 감지한 다음 각 박스 내에서 분할을 수행하는 방식입니다. 높은 정확도를 자랑하지만 속도가 느립니다.
*   **원 스테이지 인스턴스 분할**: Deep Watershed Transform, InstanceFCN, YOLACT, TensorMask, ExtremeNet 등은 더 간단한 파이프라인을 목표로 합니다.
*   **폴라 표현 활용**: [28]에서 세포 감지를 위해 폴라 표현을 사용한 선례가 있으나, 이는 더 단순한 문제였습니다. ESESeg [31]는 본 연구와 동시 진행된 연구로, 폴라 좌표계를 활용하지만 PolarMask는 다른 설계 덕분에 훨씬 우수한 성능을 달성했습니다. 대부분의 기존 방법은 인스턴스를 직접 모델링하지 않거나 최적화하기 어렵습니다.

## 🛠️ Methodology
PolarMask는 백본 네트워크, 특징 피라미드 네트워크(FPN), 그리고 두 개의 태스크별 헤드로 구성된 단순하고 통합된 네트워크입니다. FCOS [29]의 설정을 기반으로 합니다.

*   **폴라 표현 (Polar Representation)**:
    *   주어진 인스턴스 마스크에 대해, 먼저 인스턴스의 질량 중심(mass-center) $(x_c, y_c)$을 샘플링합니다.
    *   중심점으로부터 $n$개의 광선(rays)을 동일한 각도 간격 $\Delta\theta$ (예: $n=36, \Delta\theta=10^\circ$)으로 방출하여 컨투어(contour)까지의 거리를 예측합니다.
    *   인스턴스 마스크를 중심 분류와 $n$개 광선 길이 $\{d_1, d_2, \dots, d_n\}$의 밀집 거리 회귀 문제로 공식화합니다.

*   **질량 중심 (Mass Center) 선택**:
    *   바운딩 박스 중심보다 질량 중심이 인스턴스 내부에 있을 확률이 높아 마스크 예측에 더 유리함을 확인했습니다.

*   **중심 샘플 (Center Samples)**:
    *   질량 중심 주변의 $1.5 \times \text{stride}$ 영역에 해당하는 위치를 긍정적 중심 샘플로 간주합니다. 이는 긍정적 샘플 수를 늘려 불균형을 피하고, 네트워크가 최적의 인스턴스 중심을 자동으로 찾도록 돕습니다.

*   **거리 회귀 (Distance Regression)**:
    *   각 광선의 길이를 예측합니다. 여러 교차점이 있을 경우 최대 길이를, 마스크 외부에 중심이 있어 교차점이 없을 경우 최소값($10^{-6}$)을 회귀 목표로 설정합니다.
    *   이러한 코너 케이스가 이론적 상한선을 제한하지만, 실용적인 성능 향상에 초점을 맞추는 것이 더 중요하다고 주장합니다.

*   **마스크 조립 (Mask Assembling)**:
    *   추론 시, 예측된 중심 샘플 $(x_c, y_c)$와 $n$개 광선의 길이 $\{d_1, d_2, \dots, d_n\}$를 사용하여 각 컨투어 점 $(x_i, y_i)$를 다음 수식을 통해 계산합니다:
        $$ x_i = \cos\theta_i \times d_i + x_c $$
        $$ y_i = \sin\theta_i \times d_i + y_c $$
    *   $0^\circ$부터 시작하여 컨투어 점들을 연결하여 전체 컨투어 및 마스크를 조립합니다. 중복 마스크 제거를 위해 NMS(Non-Maximum Suppression)를 적용합니다.

*   **Polar Centerness**:
    *   하나의 인스턴스에 대한 $n$개 광선의 길이 $\{d_1, d_2, \dots, d_n\}$가 주어졌을 때, Polar Centerness는 다음과 같이 정의됩니다:
        $$ \text{Polar Centerness} = \sqrt{\frac{\min(\{d_1, d_2, \dots, d_n\})}{\max(\{d_1, d_2, \dots, d_n\})}} $$
    *   이는 광선 길이의 다양성(즉, 마스크의 불규칙성)이 낮을수록 높은 가중치를 부여합니다. 분류 브랜치와 병렬로 단일 레이어 브랜치를 통해 예측되며, 추론 시 분류 점수에 곱해져 저품질 마스크를 낮게 평가합니다.

*   **Polar IoU Loss**:
    *   마스크의 IoU (Intersection over Union) 정의에서 시작하여 폴라 좌표계에서 마스크 IoU를 다음과 같이 계산합니다:
        $$ \text{IoU} = \frac{\int^{2\pi}_0 \frac{1}{2}\min(d, d^\ast)^2 d\theta}{\int^{2\pi}_0 \frac{1}{2}\max(d, d^\ast)^2 d\theta} \approx \frac{\sum^n_{i=1} d_{\min}}{\sum^n_{i=1} d_{\max}} $$
        여기서 $d$는 예측된 광선 길이, $d^\ast$는 ground-truth 광선 길이이며, $d_{\min} = \min(d_i, \hat{d}_i)$이고 $d_{\max} = \max(d_i, \hat{d}_i)$ 입니다.
    *   Polar IoU Loss는 Polar IoU의 이진 교차 엔트로피(BCE) 손실이며, $\log\frac{\sum^n_{i=1} d_{\max}}{\sum^n_{i=1} d_{\min}}$로 표현됩니다.
    *   이는 미분 가능하고 병렬 계산이 용이하며, 전체 광선들을 한 번에 훈련하여 정확도를 크게 향상시키고 분류 및 회귀 손실의 균형을 자동으로 맞춰줍니다.

## 📊 Results
*   **COCO 데이터셋**으로 성능을 평가했습니다.
*   **광선 수**: 36개의 광선(AP 27.7%)이 18/24개의 광선보다 AP를 1.5%p 향상시켰습니다. 72개의 광선은 성능이 포화되는 경향을 보였습니다. 질량 중심이 바운딩 박스 중심보다 인스턴스를 더 잘 나타냅니다.
*   **Polar IoU Loss vs. Smooth-$l_1$ Loss**: Polar IoU Loss (AP 27.7%)는 smooth-$l_1$ loss (최고 AP 25.1%)보다 2.6%p 높은 AP를 달성했습니다. Polar IoU Loss는 하이퍼파라미터 튜닝 없이도 안정적인 성능을 보였으며, smooth-$l_1$ loss에서 나타나는 시스템적 오류를 줄였습니다.
*   **Polar Centerness vs. Centerness**: Polar Centerness (AP 29.1%)는 원래 Centerness보다 1.4%p AP를 향상시켰으며, 특히 엄격한 AP$_{75}$ (2.3%p↑) 및 대형 인스턴스 AP$_L$ (2.6%p↑)에서 큰 이득을 보였습니다.
*   **바운딩 박스 브랜치**: 추가 바운딩 박스 예측 브랜치는 마스크 예측 성능에 거의 영향을 미치지 않았습니다(27.7% AP vs. 27.5% AP). 따라서 간결성과 속도 향상을 위해 PolarMask에서는 이를 제거했습니다.
*   **백본 아키텍처**: ResNet-50-FPN (29.1% AP)을 기준으로, ResNet-101-FPN (30.4% AP), ResNeXt-101-FPN (32.6% AP) 등 더 깊고 발전된 백본을 사용할수록 성능이 향상되었습니다.
*   **속도와 정확도 트레이드오프**: ResNet-50 백본과 800px 짧은 변 스케일에서 29.1% AP와 17.2 FPS (V100 GPU)를 달성하여 실시간 인스턴스 분할 애플리케이션으로의 잠재력을 보여주었습니다.
*   **최신 기술과의 비교 (COCO test-dev)**: ResNeXt-101-FPN과 12 에폭 훈련 시 32.9% mask AP를 달성했습니다. 변형 가능한 합성곱(DCN)을 추가하고 24 에폭 훈련 시 36.2% mask AP를 달성하여 최신 투 스테이지 및 원 스테이지 방법들과 경쟁력 있는 성능을 보였습니다. TensorMask보다 4.7배 빠르고, YOLACT보다 0.9%p 높은 AP를 달성했습니다.
*   **ESE-Seg와의 비교**: Polar IoU Loss와 Polar Centerness 덕분에 ESE-Seg보다 7.5%p 높은 AP를 달성했으며, 박스 감지가 필요 없는 점이 차이점입니다.

## 🧠 Insights & Discussion
*   **단순성과 효율성**: PolarMask는 인스턴스 분할이 바운딩 박스 객체 감지만큼 단순하고 계산적으로 가벼울 수 있음을 증명하며, 단일 샷 인스턴스 분할의 강력한 기본 모델이 될 수 있습니다.
*   **폴라 표현의 장점**: 폴라 좌표계는 마스크 컨투어를 표현하는 자연스러운 방법을 제공하며, 마스크 예측 문제를 중심 분류 및 거리 회귀로 단순화합니다. 방향성을 가진 각도는 컨투어 연결을 용이하게 합니다.
*   **혁신적인 구성 요소**: Polar IoU Loss와 Polar Centerness는 밀집 거리 회귀와 고품질 샘플 선택이라는 폴라 표현의 주요 과제를 효과적으로 해결하여 성능 향상과 최적화 용이성을 가져왔습니다.
*   **유연성**: PolarMask 프레임워크는 FCOS와 같은 기존 단일 샷 객체 감지기에 최소한의 수정으로 쉽게 통합될 수 있는 높은 유연성을 가집니다.
*   **제한 사항**: 광선이 컨투어와 여러 번 교차하거나 전혀 교차하지 않는 등 일부 코너 케이스가 이론적으로 100% AP 달성을 제한할 수 있지만, 실제 모델 성능 향상에 집중하는 것이 더 중요하다고 지적합니다.

## 📌 TL;DR
**문제**: 인스턴스 분할은 복잡하고 느리며, 특히 단일 샷 방법에서는 성능 저하가 흔했습니다.
**방법**: PolarMask는 앵커 박스 프리 단일 샷 인스턴스 분할 방법으로, 마스크를 폴라 좌표계의 중심점 분류와 광선 거리 회귀로 표현합니다. 밀집 거리 회귀를 위한 Polar IoU Loss와 고품질 샘플 가중치를 위한 Polar Centerness를 도입합니다.
**결과**: PolarMask는 COCO 데이터셋에서 경쟁력 있는 mask mAP (예: ResNeXt-101로 32.9%)를 달성했으며, 기존 방법보다 더 단순하고 빠르며 완전 합성곱(fully convolutional) 방식으로 인스턴스 분할이 객체 감지처럼 간단해질 수 있음을 입증했습니다.
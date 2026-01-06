# Style Normalization and Restitution for Domain Generalization and Adaptation

Xin Jin, Cuiling Lan, Wenjun Zeng, Zhibo Chen

## 🧩 Problem to Solve

학습된 딥러닝 모델은 학습 데이터와 다른 환경(예: 조명, 색상 대비 등)에서 획득된 테스트 이미지에 대해 현저한 성능 저하를 겪는다는 문제가 있습니다. 특히, 도메인 일반화(Domain Generalization, DG) 및 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)과 같은 시나리오에서, 모델이 보지 못했던 도메인에 대해 일반화 및 판별 능력을 동시에 유지하는 것은 어려운 과제입니다. Instance Normalization (IN)은 스타일 변이를 줄여 일반화 능력을 향상시키지만, 이 과정에서 태스크 관련 판별 정보가 손실되어 성능 저하를 초래할 수 있습니다.

## ✨ Key Contributions

- **Style Normalization and Restitution (SNR) 모듈 제안**: 기존 신경망에 쉽게 적용 가능한 플러그앤플레이(plug-and-play) 모듈인 SNR을 통해 네트워크의 일반화 능력을 강화합니다. 스타일 정규화로 인한 판별 정보 손실을 보상하기 위해, 원본 피처와 IN 피처 간의 잔차(residual)에서 태스크 관련 판별 정보를 추출하여 복원하는 방식을 제안합니다.
- **이중 복원 손실(Dual Restitution Loss) 제안**: 잔차 정보에서 태스크 관련 피처와 태스크 무관 피처를 더 잘 분리(disentanglement)하도록 돕는 이중 복원 손실 제약을 도입합니다.
- **다양한 비전 태스크에 대한 일반성 및 성능 향상**: 제안된 SNR 모듈은 객체 분류, 검출, 의미론적 분할 등 다양한 컴퓨터 비전 태스크에 적용되어 네트워크의 일반화 능력을 향상시킬 수 있음을 입증합니다. 또한, 기존 UDA 네트워크의 성능도 개선할 수 있습니다.
- **확장된 연구**: 기존 컨퍼런스 논문 [37]이 특정 작업(person re-identification)에 초점을 맞춘 데 반해, 본 연구에서는 SNR 디자인을 일반화하여 다양한 컴퓨터 비전 태스크에 적용하고 엔트로피 비교를 활용하여 이중 복원 손실을 맞춤화합니다.

## 📎 Related Works

- **Domain Generalization (DG)**: 타겟 도메인 데이터에 접근 없이 모델을 일반화하는 방법으로, 메타 학습(Meta-learning) [7], 에피소딕 훈련(Episodic training) [9], 적응적 앙상블 학습(Adaptive ensemble learning) [31] 등이 있습니다. 또한, Instance Normalization (IN)을 CNN에 통합하여 모델 일반화 능력을 향상시키려는 연구 [29, 34]도 있습니다. 하지만 IN은 판별 정보 손실을 유발하는 단점이 있습니다.
- **Unsupervised Domain Adaptation (UDA)**: 레이블된 소스 도메인과 레이블 없는 타겟 도메인 데이터를 활용하여 도메인 간의 분포 차이를 줄여 도메인 불변 피처를 학습하는 방법입니다. MMD (Maximum Mean Discrepancy) [12], CORAL (CORrelation ALignment) [18], Adversarial Learning (DANN [24]) 등이 주로 사용됩니다.
- **Feature Disentanglement**: 관련 없는 피처를 제거하여 학습된 표현을 분리하는 데 중점을 둡니다 [46, 47, 48]. 본 논문은 이러한 아이디어에서 영감을 받아, 손실된 판별 정보를 복원하기 위해 잔차 피처에서 태스크 고유 피처를 분리합니다.

## 🛠️ Methodology

제안된 Style Normalization and Restitution (SNR) 모듈은 기존 백본 네트워크의 컨볼루션 블록 뒤에 플러그앤플레이 방식으로 삽입됩니다 (예: ResNet-50).

1. **스타일 정규화(Style Normalization)**:

   - 입력 피처 맵 $F \in \mathbb{R}^{h \times w \times c}$에 대해 Instance Normalization (IN)을 수행하여 샘플/인스턴스 간의 스타일 불일치(discrepancy)를 제거합니다. IN은 각 채널의 공간 차원에 걸쳐 평균과 표준 편차를 계산하여 정규화합니다.
   - $$ \tilde{F} = \text{IN}(F) = \gamma \left( \frac{F - \mu(F)}{\sigma(F)} \right) + \beta $$
   - $\mu(\cdot)$와 $\sigma(\cdot)$는 각각 평균과 표준 편차를 나타내며, $\gamma, \beta \in \mathbb{R}^c$는 학습 가능한 파라미터입니다.
   - IN은 스타일 정보를 제거하여 일반화 능력을 향상시키지만, 태스크 관련 판별 정보도 손실시킬 수 있습니다.

2. **피처 복원(Feature Restitution)**:

   - IN으로 인해 손실된 판별 정보를 보존하기 위해, 원본 피처 $F$와 스타일 정규화된 피처 $\tilde{F}$의 차이인 잔차 $R = F - \tilde{F}$에서 태스크 관련 피처를 추출하고 이를 네트워크에 다시 추가합니다.
   - 잔차 $R$을 채널 어텐션 응답 벡터 $a \in \mathbb{R}^c$를 통해 태스크 관련 피처 $R_+$와 태스크 무관 피처 $R_-$로 분리합니다:
     $$ R*+(:, :, k) = a_k R(:, :, k) $$
        $$ R*-(:, :, k) = (1 - a_k) R(:, :, k) $$
        여기서 $a = g(R) = \sigma(\text{W}_2 \delta(\text{W}_1 \text{pool}(R)))$로 계산됩니다.
   - 추출된 태스크 관련 피처 $R_+$를 스타일 정규화된 피처 $\tilde{F}$에 더하여 최종 출력 피처 $\tilde{F}_+ = \tilde{F} + R_+$를 얻습니다.
   - 태스크 무관 피처 $R_-$를 $\tilde{F}$에 더한 오염된 피처 $\tilde{F}_- = \tilde{F} + R_-$는 이중 복원 손실 계산에 사용됩니다.

3. **이중 복원 손실 제약(Dual Restitution Loss Constraint)**:
   - 태스크 관련 피처와 태스크 무관 피처의 분리(disentanglement)를 촉진하기 위해 제안됩니다.
   - 향상된 피처 $\tilde{F}_+$는 정규화된 피처 $\tilde{F}$보다 더 판별적(discriminative)이어야 하므로, 예측 클래스 확률의 엔트로피가 작아져야 합니다.
   - 오염된 피처 $\tilde{F}_-$는 $\tilde{F}$보다 덜 판별적이어야 하므로, 예측 클래스 확률의 엔트로피가 커져야 합니다.
   - 분류 태스크의 경우:
     $$ L*{SNR,+} = \text{Softplus}(H(\phi(\tilde{f}*+)) - H(\phi(\tilde{f}))) $$
        $$ L*{SNR,-} = \text{Softplus}(H(\phi(\tilde{f})) - H(\phi(\tilde{f}*-))) $$
        여기서 $H(\cdot)$는 엔트로피 함수, $\phi(\cdot)$는 FC 레이어와 소프트맥스를 나타내고, $\tilde{f}_+, \tilde{f}, \tilde{f}_-$는 각 피처 맵을 공간 평균 풀링한 피처 벡터입니다.
   - 의미론적 분할 및 객체 검출 태스크에서는 픽셀 또는 바운딩 박스 영역별 엔트로피를 계산하여 손실을 적용합니다.

## 📊 Results

- **도메인 일반화(DG) (객체 분류)**:
  - PACS 및 Office-Home 데이터셋에서 기존 SOTA 방법들을 능가하며, Baseline (AGG) 대비 PACS에서 2.3%, Office-Home에서 1.4%의 평균 정확도 향상을 달성했습니다.
  - L2A-OT [55]는 데이터 생성기로 소스 도메인을 증강하여 PACS에서 최고의 성능을 보였는데, SNR은 IN을 통한 내재적인 스타일 정규화로 개념적으로 보완될 수 있다고 언급됩니다.
- **비지도 도메인 적응(UDA) (객체 분류)**:
  - Digit-Five 및 mini-DomainNet 데이터셋에서 SNR-M3SDA는 SOTA인 M3SDA-$\beta$ [15]를 Digit-Five에서 6.47%, mini-DomainNet에서 2.03% 유의미하게 능가했습니다.
  - 풀 DomainNet 데이터셋에서도 SNR-M3SDA는 Baseline (M3SDA) 대비 4.0% 향상된 46.67%의 평균 정확도를 달성했습니다.
- **의미론적 분할(Semantic Segmentation)**:
  - GTA5-to-Cityscapes 및 Synthia-to-Cityscapes 설정에서 DG 및 UDA 모두 Baseline 및 Baseline-IN 대비 mIoU에서 상당한 개선을 보였습니다 (예: DeeplabV2 백본에서 GTA5-to-Cityscapes로 DG 시 5.74% mIoU 향상).
  - UDA 설정에서는 SNR-MCD와 SNR-MS가 각각 MCD [13] 및 MaxSquare (MS) [71] 대비 mIoU에서 5.3% 및 2.2% 이상 향상되었습니다.
- **객체 검출(Object Detection)**:
  - Cityscapes-to-Foggy Cityscapes (날씨 변화) 및 Cross-Dataset (Cityscapes-KITTI) 시나리오에서 Faster R-CNN [52] 및 DA Faster R-CNN [72] 대비 mAP에서 3.0-5.7% 향상을 기록했습니다.
- **복잡도 분석**: SNR 모듈은 ResNet-50 백본에 대해 모델 크기 2.2%, FLOPs 5.1%의 작은 증가만을 가져와 효율성을 입증했습니다.

## 🧠 Insights & Discussion

- **일반화 및 판별 능력 동시 강화**: SNR은 IN을 통한 스타일 정규화로 일반화 능력을 높이고, 잔차 정보에서 태스크 관련 피처를 복원하여 판별 능력을 보존하는 효과적인 방법입니다. 이 두 가지 목표를 동시에 달성함으로써 다양한 도메인 간의 성능 저하 문제를 완화합니다.
- **이중 복원 손실의 중요성**: 엔트로피 비교를 기반으로 하는 이중 복원 손실은 태스크 관련/무관 피처의 효과적인 분리를 유도하며, 이는 최종 성능 향상에 크게 기여합니다. 비교 방식이 단순히 엔트로피를 최소화/최대화하는 것보다 더 우수한 결과를 보였습니다.
- **모듈의 유연성**: SNR은 플러그앤플레이 모듈로서 다양한 백본 네트워크 및 컴퓨터 비전 태스크(분류, 분할, 검출)에 쉽게 통합될 수 있으며, 일관된 성능 향상을 제공합니다.
- **낮은 추가 복잡도**: SNR 모듈 추가로 인한 계산 복잡도 및 모델 크기 증가는 미미하여, 실용적인 적용 가능성이 높습니다.
- **시각화 분석**: 활성화 맵 시각화는 SNR이 태스크 관련 객체 피처를 더 잘 분리하고, 스타일 변화에 대해 더 일관된 응답을 보임을 확인시켜 줍니다. t-SNE 시각화는 SNR이 클래스 간의 피처를 더 명확하게 분리함을 보여줍니다.
- **한계 및 향후 연구**: 본 연구는 주로 스타일 정규화와 판별 정보 복원에 초점을 맞추었으며, L2A-OT [55]와 같이 데이터 증강을 통해 입력 다양성을 높이는 방법과 개념적으로 상호 보완적입니다. SNR을 이러한 데이터 증강 기법과 결합하면 추가적인 성능 향상을 기대할 수 있습니다.

## 📌 TL;DR

본 논문은 딥러닝 모델의 도메인 일반화 및 적응 능력 향상을 위해 **Style Normalization and Restitution (SNR)** 모듈을 제안합니다. SNR은 Instance Normalization으로 스타일 변이를 제거하여 일반화 능력을 높이고, 이 과정에서 손실되는 태스크 관련 판별 정보를 잔차에서 추출하여 복원합니다. 또한, 태스크 관련/무관 피처의 효과적인 분리를 위한 **이중 복원 손실**을 도입합니다. 객체 분류, 의미론적 분할, 객체 검출 등 다양한 컴퓨터 비전 태스크에서 SOTA 성능을 달성하며, 낮은 추가 복잡도로 모델의 일반화 및 판별 능력을 동시에 강화함을 입증합니다.

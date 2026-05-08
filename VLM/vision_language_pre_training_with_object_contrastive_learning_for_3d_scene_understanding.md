# Vision-Language Pre-training with Object Contrastive Learning for 3D Scene Understanding

Taolin Zhang, Sunan He, Tao Dai, Bin Chen, Zhi Wang and Shu-Tao Xia (2023)

## 🧩 Problem to Solve

본 논문은 3D 장면 이해(3D Scene Understanding)를 위한 시각-언어 사전 학습(Vision-Language Pre-training, VLP) 프레임워크의 부재와 그로 인한 일반화 성능 저하 문제를 해결하고자 한다.

현재 3D 시각-언어 관련 연구들은 주로 특정 작업(Task-specific)에 최적화된 모델을 구축하는 데 집중하고 있으며, 이는 다음과 같은 한계를 가진다.
첫째, 각 작업에서 얻어진 표현(Representation)이 다른 작업으로 잘 전이되지 않아 범용적인 3D 시각-언어 임베딩을 추출하지 못한다.
둘째, 3D 포인트 클라우드(Point Cloud) 데이터는 2D 이미지나 텍스트와는 다른 특성을 가지므로, 기존의 2D VLP나 NLP 사전 학습 목적 함수를 그대로 적용하기 어렵다.
셋째, 3D 장면 내의 유사한 객체들을 구분하고, 포인트 클라우드 특징과 언어 토큰 임베딩 간의 서로 다른 모달리티 공간을 정렬(Alignment)하는 것이 매우 까다롭다.

따라서 본 연구의 목표는 3D 시각-언어 하위 작업들에 유연하게 전이될 수 있는 범용적인 사전 학습 프레임워크인 3DVLP를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 시각-언어 작업들의 공통적인 특성(객체 탐지 의존성 및 융합 모듈의 필요성)을 분석하여, 이를 최적화할 수 있는 세 가지 객체 수준(Object-level)의 프록시 태스크(Proxy Task)를 설계하는 것이다.

1. **Object-level IoU-guided Detection (OID)**: 고품질의 바운딩 박스(Bounding Box) 제안을 생성하여 하위 작업의 기초 성능을 높인다.
2. **Object-level Cross-Contrastive alignment (OCC)**: 포인트 클라우드의 객체 제안과 언어 설명 간의 분포를 정렬하여 모달리티 간 상호작용을 강화한다.
3. **Object-level Self-Contrastive learning (OSC)**: 동일 장면 내의 서로 다른 객체들을 구분하는 능력을 배양하여 모델의 세밀한 의미 이해도를 높인다.

## 📎 Related Works

### 2D Vision-Language Pre-training

CLIP, ALBEF, UNITER 등 2D 분야에서는 대규모 이미지-텍스트 쌍을 이용한 사전 학습이 큰 성과를 거두었다. 주로 대조 학습(Contrastive Loss)이나 마스킹 언어 모델링(MLM)을 통해 모달리티 간 정렬을 수행한다. 그러나 이러한 방식은 3D 포인트 클라우드의 기하학적 특성을 반영하지 못해 직접 적용이 불가능하다.

### 3D Visual-Language Tasks

3D Visual Grounding, 3D Dense Captioning, 3D Question Answering 등이 연구되어 왔다. 대부분의 기존 방법론은 객체 탐지 후 매칭하는 2단계 파이프라인(Detection-then-match)을 사용하며, Cross-attention 모듈을 통해 융합을 시도한다. 3DJCG나 D3Net 같은 연구가 일부 작업 간의 공동 학습을 시도했으나, 범용적인 사전 학습 모델을 통해 하위 작업의 성능을 높이려 한 시도는 본 연구가 처음이다.

## 🛠️ Methodology

### 전체 시스템 구조

3DVLP는 포인트 클라우드 인코더(VoteNet)와 언어 인코더(Frozen BERT)를 사용하여 각각의 특징을 추출하고, Cross-attention 모듈을 통해 융합 특징을 생성한다. 학습은 사전 학습(Pre-training) 단계와 하위 작업별 미세 조정(Fine-tuning) 단계로 나뉜다.

### 1. Object-level IoU-guided Detection (OID) Loss

고품질의 객체 제안을 얻기 위해 Visual Grounding을 프록시 태스크로 설정하고, DIoU(Distance IoU) 손실과 Label Smoothing을 결합한 OID 손실을 도입한다.

- **DIoU Loss**: 예측 박스 $b^p$와 정답 박스 $b^{gt}$ 사이의 중심점 거리 $\rho$와 최소 외접 박스의 대각선 길이 $c$를 고려하여 계산한다.
$$L_{DIoU}(b^p, b^{gt}) = 1 - IoU + \frac{\rho^2(b^p, b^{gt})}{c^2}$$
- **Label Smoothing**: IoU 필터 임계값 $\tau$를 사용하여, $IoU \ge \tau$인 제안들을 양성 샘플로 간주하고 가중치 $y_p$를 부여함으로써 모델의 과잉 확신을 방지한다.
- **최종 OID 손실**:
$$L_{OID} = \sum_p y_p \cdot L_{DIoU}(b^p, b^{gt})$$

### 2. Object-level Cross-Contrastive alignment (OCC)

모달리티 간의 정렬을 위해 객체 제안 특징 $\mathcal{R}_p$와 언어 임베딩 $\mathcal{T}$ 간의 대조 학습을 수행한다. IoU 필터를 통해 선택된 양성 샘플 집합 $\mathcal{P}_{pos}$는 언어 임베딩과 가깝게, 음성 샘플 집합 $\mathcal{P}_{neg}$는 멀게 밀어낸다.

$$L_{OCC} = -\frac{1}{2} \mathbb{E}_{(b^{gt}, \mathcal{T}) \sim \mathcal{D}} \left[ \log \frac{\sum_{p \in \mathcal{P}_{pos}} \exp(s(\mathcal{R}_p, \mathcal{T}))}{\sum_{\hat{p} \in \mathcal{P}_{pos} \cup \mathcal{P}_{neg}} \exp(s(\mathcal{R}_{\hat{p}}, \mathcal{T}))} + \log \frac{\sum_{p \in \mathcal{P}_{pos}} \exp(s(\mathcal{T}, \mathcal{R}_p))}{\sum_{\hat{p} \in \mathcal{P}_{pos} \cup \mathcal{P}_{neg}} \exp(s(\mathcal{T}, \mathcal{R}_{\hat{p}}))} \right]$$
여기서 $s(\cdot, \cdot)$는 내적(dot product)과 같은 유사도 함수이다.

### 3. Object-level Self-Contrastive learning (OSC)

포인트 클라우드 내에서 객체 간의 변별력을 높이기 위해 단일 모달리티 내에서 대조 학습을 수행한다. 언어 임베딩 대신 다른 객체 제안들의 임베딩을 사용하여, 정답 객체와 겹치는 제안들은 서로 가깝게, 그렇지 않은 제안들은 멀게 배치한다.

$$L_{OSC} = -\mathbb{E}_{b^{gt} \sim \mathcal{D}} \left[ \log \frac{\sum_{p, \hat{p} \in \mathcal{P}_{pos}} \exp(s(\mathcal{R}_p, \mathcal{R}_{\hat{p}}))}{\sum_{p, \hat{p} \in \mathcal{P}_{pos} \cup \mathcal{P}_{neg}} \exp(s(\mathcal{R}_p, \mathcal{R}_{\hat{p}}))} \right]$$

### 하위 작업별 헤드 (Downstream Heads)

- **3D Visual Grounding**: 융합 특징을 사용하여 $n$개의 제안 중 정답을 찾는 분류 문제로 정의하며 Cross-entropy loss를 사용한다.
- **3D Dense Captioning**: Transformer 디코더를 사용하여 객체별 설명을 생성하며, Cross-entropy loss와 Masked Language Modeling(MLM)을 사용한다.
- **3D Question Answering**: 가능한 모든 답변 후보군에 대해 다중 클래스 분류 문제로 정의하며, MLP 헤드를 통해 최종 답변을 예측한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanRefer (Grounding), Scan2Cap (Captioning), ScanQA (QA)
- **평가 지표**: IoU@0.25/0.5 (Grounding), CIDEr, BLEU-4, METEOR, ROUGE (Captioning), EM@1/10 (QA)
- **구현 상세**: VoteNet(포인트 클라우드 인코더), Frozen BERT(언어 인코더), AdamW 옵티마이저 사용.

### 주요 결과

1. **3D Visual Grounding**: 3DVLP-2D+3D 모델은 Acc@0.5 기준 40.51%를 달성하여 기존 SOTA 모델들을 크게 상회하였다. 특히 OID 손실이 고품질 바운딩 박스 생성에 기여했음을 확인하였다.
2. **3D Dense Captioning**: 사전 학습된 인코더의 범용적 특징 추출 능력을 바탕으로, METEOR@0.5 지표에서 기존 모델 대비 상당한 성능 향상을 보였다. 이는 특히 의미적 유사성과 유창성 부분에서 강점이 있음을 시사한다.
3. **3D Question Answering**: EM@1 기준 24.03%를 기록하며 기존 ScanQA 및 FE-3DGQA보다 우수한 성능을 보였다.

### Ablation Study 및 분석

- **모듈별 기여도**: OID는 바운딩 박스의 품질을 높여 Acc@0.5를 크게 향상시켰으며, OCC와 OSC는 복잡한 장면 내에서 객체를 정확히 구분하고 매칭하는 성능을 높였다.
- **임계값 $\tau$ 영향**: IoU 필터의 임계값이 너무 낮으면 잘못된 샘플이 양성으로 간주되어 최적화를 방해하고, 너무 높으면 학습 데이터 부족으로 성능이 하락한다. 실험 결과 $\tau=0.25$가 가장 적절한 트레이드오프 지점임이 밝혀졌다.
- **사전 학습 효과**: 처음부터 학습(Train from scratch)한 모델과 비교했을 때, 3DVLP의 사전 학습 단계가 캡셔닝 지표에서 0.5~6%, QA 지표에서 2~4%의 성능 향상을 가져왔다.

## 🧠 Insights & Discussion

본 논문은 3D 장면 이해의 핵심이 **'정교한 객체 제안'**, **'모달리티 간 정렬'**, **'객체 간 변별력'**에 있다는 점을 꿰뚫어 보고 이를 세 가지 프록시 태스크(OID, OCC, OSC)로 구체화하였다.

특히 인상적인 점은 단순히 데이터 양을 늘리는 것이 아니라, IoU 필터를 활용한 객체 수준의 대조 학습을 통해 3D 데이터의 특수성을 해결하려 했다는 점이다. t-SNE 시각화 결과에서도 3DVLP가 기본 모델보다 객체별 특징을 훨씬 더 명확하게 군집화하는 것을 확인할 수 있어, 제안한 방법론이 유효했음을 입증한다.

다만, 논문에서 언급되었듯이 포인트 클라우드와 언어 간의 완전한 상호작용을 위한 다층적 정보 융합(multi-level information interaction) 부분은 향후 과제로 남아 있으며, 현재의 융합 방식이 최적의 구조인지에 대한 추가적인 논의가 필요해 보인다.

## 📌 TL;DR

3DVLP는 3D 장면 이해를 위한 새로운 시각-언어 사전 학습 프레임워크이다. 이 모델은 고품질 박스 생성을 위한 **OID**, 모달리티 정렬을 위한 **OCC**, 객체 구분 능력을 위한 **OSC**라는 세 가지 객체 수준의 프록시 태스크를 도입하였다. 이를 통해 3D Visual Grounding, Dense Captioning, QA라는 서로 다른 세 가지 하위 작업 모두에서 SOTA 성능을 달성하며, 3D VLP의 범용적인 임베딩 학습 가능성을 증명하였다.

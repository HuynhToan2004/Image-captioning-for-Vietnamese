import argparse
from cmath import nan
import os
import json
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, BartTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration
from torchvision import models, transforms
import types
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=str, default="684331")
parser.add_argument("--gpu_ids", type=str, default="0")  
parser.add_argument("--num_workers", type=int, default=4)

parser.add_argument("--article_max_length", type=int, default=512)
parser.add_argument("--caption_max_length", type=int, default=100)
parser.add_argument("--plm_type", type=str, default="/data/npl/ICEK/VACNIC/src/data/assest/mbart-large-cc25")

parser.add_argument("--clip_type", type=str, default="ViT-B-32") 
parser.add_argument("--ent_start_token", type=str, default="no")
parser.add_argument("--ent_end_token", type=str, default="no")

parser.add_argument("--enc_fusion_layer", type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11], help="Fusion layers in encoder")
parser.add_argument("--dim_common", type=int, default=768)

parser.add_argument("--warmup_rate", type=float, default=0.05)
parser.add_argument("--train_batch_size", type=int, default=1)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument("--beam_size", type=int, default=1)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--num_epoch", type=int, default=2)
parser.add_argument("--lr_bart", type=float, default=1e-4)
parser.add_argument("--lr_clip", type=float, default=5e-6)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--clip_norm", type=float, default=0.1)

# Đổi default data_type thành "vietnamese"
parser.add_argument("--data_type", type=str, default="vietnamese")
parser.add_argument("--data_dir", type=str, default="/data/npl/ICEK/VACNIC/data/train")
parser.add_argument("--out_dir", type=str, default="/data/npl/ICEK/VACNIC/output")

parser.add_argument("--mapping_loss_type", type=str, default="contrastive")

parser.add_argument("--trained_clip", type=str, default="no")
parser.add_argument("--clip_dir", type=str, default=".")
parser.add_argument("--no_clip_loss", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--prompt_size", type=int, default=10)
parser.add_argument("--use_vis_cls", default=False, type=lambda x: (str(x).lower() == 'true'))

# parser.add_argument("--freeze_layer", nargs="+", type=int)
parser.add_argument("--max_ner_type_len", type=int, default=80)
parser.add_argument("--max_ner_type_len_gt", type=int, default=20)

parser.add_argument("--freeze_clip", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--prompt_mlp_type", type=str, default="clipcap")
parser.add_argument("--map_size", nargs="+", type=int)

parser.add_argument("--no_mapping", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--mapping_loss_weight", type=float, default=1.0)
parser.add_argument("--img_size", type=int, default=512)

parser.add_argument("--only_image", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--use_secla", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--num_sentences", type=int, default=8)

parser.add_argument("--adapter_dim", type=int, default=96)
parser.add_argument("--project_name", type=str, default="news_cap")
parser.add_argument("--experiment_name", type=str, default="t5_retrieval_goodnews")

# Không cần wandb nữa nên xóa offline_wandb
parser.add_argument("--offline_wandb", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--perturb", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--no_clip_norm", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--init_attn_weight", default=False, type=lambda x: (str(x).lower() == 'true'))

parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--alpha", type=float, default=1.0)

args = parser.parse_args()


def prep_for_training(model, train_size, DEVICE):
    model.to(DEVICE)
    optimizer_bart = optim.AdamW(
        list(model.model.parameters()) + list(model.lm_head.parameters()),
        betas=(0.9, 0.999), lr=args.lr_bart, eps=1e-8, weight_decay=args.weight_decay
    )
    optimizer_clip = optim.AdamW(
        list(model.clip_img_model.parameters()) + list(model.clip_txt_model.parameters()),
        betas=(0.9, 0.999), lr=args.lr_clip, eps=1e-8, weight_decay=args.weight_decay
    )

    num_training_steps = args.num_epoch * train_size / args.train_batch_size
    num_warmup_steps = int(args.warmup_rate * num_training_steps)

    scheduler_bart = get_linear_schedule_with_warmup(optimizer_bart, num_warmup_steps, num_training_steps)
    scheduler_clip = get_linear_schedule_with_warmup(optimizer_clip, num_warmup_steps, num_training_steps)

    return model, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip


def get_embedding_ner(model, ner_ids_3d):
    bsz, num_ner, id_len = ner_ids_3d.size()
    hidden_states_ner_list = []
    with torch.no_grad():
        # Luôn dùng branch 1 GPU
        encoder = model.model.encoder
        for i in range(num_ner):
            ner_ids = ner_ids_3d[:, i, :].squeeze(1)
            ner_shape = ner_ids.size()
            ner_ids = ner_ids.view(-1, ner_shape[-1])
            ner_embeds = encoder.embed_tokens_ner(ner_ids) * encoder.embed_scale
            embed_pos_ner = encoder.embed_positions_ner(ner_shape)
            hidden_states_ner = ner_embeds + embed_pos_ner
            hidden_states_ner = encoder.layernorm_embedding_ner(hidden_states_ner)
            hidden_states_ner = torch.nn.functional.dropout(hidden_states_ner, p=encoder.dropout, training=False)
            hidden_states_ner_list.append(torch.mean(hidden_states_ner, dim=1))
            del hidden_states_ner
    return torch.stack(hidden_states_ner_list, dim=1)


def get_embedding_tgt(model, tgt_ids):
    with torch.no_grad():
        decoder = model.model.decoder
        tgt_shape = tgt_ids.size()
        tgt_ids = tgt_ids.view(-1, tgt_shape[-1])
        tgt_ids = tgt_ids.cuda()
        tgt_embeds = decoder.embed_tokens(tgt_ids) * decoder.embed_scale
        embed_pos = decoder.embed_positions(tgt_shape)
        hidden_states = tgt_embeds + embed_pos
        hidden_states = decoder.layernorm_embedding(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=decoder.dropout, training=False)
    return hidden_states


def find_first_sublist(seq, sublist, start=0):
    length = len(sublist)
    for index in range(start, len(seq)):
        if seq[index: index+length] == sublist:
            return index, index+length


def get_hidden_states_ner(model, src_ids, src_mask, tgt_input, img_feat, name_ids, name_mask, org_ids, org_mask, gpe_ids, gpe_mask, ner_mask):
    with torch.no_grad():
        model_inner = model.model  
        output = model_inner(
            input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input,
            image_features=img_feat, name_ids=name_ids, org_ids=org_ids, gpe_ids=gpe_ids,
            name_mask=name_mask, org_mask=org_mask, gpe_mask=gpe_mask, ner_mask=ner_mask,
            add_img_ner_attn=False
        )["hidden_states_ner"]
    return output


def pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    emb = torch.nan_to_num(emb, nan=1.0)
    return emb


def pool_replace(last_hidden_states, attention_mask, img_feat_map):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    for i in range(emb.size()[0]):
        if torch.isnan(emb[i][0]):
            emb[i] = img_feat_map[i].detach().cpu().to(emb.device)
    return emb


def shift_tokens_right(input_ids, pad_token_id: int, decoder_start_token_id: int):
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


def create_src_mask_bart(input_ids):
    src_padding_mask = (input_ids == 1)
    src_mask = (src_padding_mask < 1)
    src_mask = src_mask.int().type(torch.int64)
    src_mask = src_mask.to(input_ids.device)
    return src_mask



import torch
from PIL import Image

def extract_clip_img_feat(clip_model, x):
    """
    x: torch.Tensor [batch_size, 3, H, W], đã ở trên GPU (DEVICE).
    Trả về (patch_emb, cls_emb) nhưng thực chất đều cùng 1 embedding cuối.
    """
   
    images_pil = []
    for i in range(x.shape[0]):
        img_arr = x[i].detach().cpu().permute(1, 2, 0).numpy()
        if img_arr.max() <= 1.0:
            img_arr = (img_arr * 255).astype("uint8")
        pil_img = Image.fromarray(img_arr.astype("uint8"))
        images_pil.append(pil_img)
    with torch.no_grad():
        img_emb = clip_model.encode(images_pil, convert_to_tensor=True)
        img_emb = img_emb.to(x.device)
    return img_emb, img_emb

def decode_and_encode_text(clip_tokenizer, clip_model, tgt_ids_clip, device):
    """
    - tgt_ids_clip: [batch_size, seq_len], token IDs do clip_tokenizer sinh ra.
    - clip_tokenizer: tokenizer của SentenceTransformer (thực tế là XLMR/BERT tokenizer).
    Trả về text_emb: tensor [batch_size, emb_dim].
    """
    text_list = []
    for row_ids in tgt_ids_clip:
        ids_cpu = row_ids.detach().cpu().numpy().tolist()
        text_str = clip_tokenizer.decode(ids_cpu, skip_special_tokens=True)
        text_list.append(text_str)

    with torch.no_grad():
        text_emb = clip_model.encode(text_list, convert_to_tensor=True)
    return text_emb.to(device)

def train_epoch(
    bart_model, model,
    loss_margin, loss_fn,
    loss_img_clip, loss_txt_clip, loss_clip_bart,
    train_dataloader,
    optimizer_bart, optimizer_clip,
    scheduler_bart, scheduler_clip,
    epoch, DEVICE
):
    model.train()
    tr_loss = 0
    tr_txt_loss = 0
    tr_clip_loss = 0
    tr_face_name_loss = 0
    tr_ner_loss = 0
    nb_tr_steps = 0
    tr_margin_loss = 0

    bi_contras_loss = BatchSoftmax()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

        src_ids, tgt_ids, tgt_ids_clip, img_tensors, face_emb, names_art_ids, names_ids, names_ids_flatten = (
            batch["article_ids"],
            batch["caption_ids"],
            batch["caption_ids_clip"],
            batch["img_tensor"],
            batch["face_emb"],
            batch["names_art_ids"],
            batch["names_ids"],
            batch["names_ids_flatten"]
        )

        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)
        names_ids_3d = names_ids.to(DEVICE)
        names_ids_flatten = names_ids_flatten.to(DEVICE)

        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)

        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)
        names_cap_mask = create_src_mask_bart(names_ids_flatten)

        img_feat, img_feat_cls = extract_clip_img_feat(model.clip_img_model, img_tensors)

        # Forward mô hình
        if args.prompt_mlp_type == "clipcap":
            output = model(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=tgt_input,
                image_features=img_feat_cls,
                face_features=face_emb,
                face_mask=face_mask,
                name_ids=names_art_ids,
                name_mask=names_art_mask,
                add_ner_ffn=True,
                return_dict=True
            )
        else:
            output = model(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=tgt_input,
                image_features=img_feat,
                face_features=face_emb,
                face_mask=face_mask,
                name_ids=names_art_ids,
                name_mask=names_art_mask,
                add_ner_ffn=True,
                return_dict=True
            )

        # Cross-entropy loss cho text
        logits = output["logits"]
        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
        tr_txt_loss += txt_loss.item()

        # So sánh hidden states với Bart gốc => margin loss
        decoder_hidden_states = output["decoder_hidden_states"][-1]

    #################################################################################################################
        # output_bart = bart_model(input_ids=src_ids, attention_mask=src_mask, decoder_input_ids=tgt_input)
        # decoder_hidden_states_bart = output_bart["decoder_hidden_states"][-1]
    #################################################################################################################
        output_bart = bart_model(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=tgt_input,
            output_hidden_states=True, 
            return_dict=True           
        )
        decoder_hidden_states_bart = output_bart["decoder_hidden_states"][-1]
        tgt_mask = create_src_mask_bart(tgt_ids)

        decoder_hidden_states = pool(decoder_hidden_states, tgt_mask)
        decoder_hidden_states_bart = pool(decoder_hidden_states_bart, tgt_mask)

        decoder_hidden_states = decoder_hidden_states / decoder_hidden_states.norm(dim=1, keepdim=True)
        decoder_hidden_states_bart = decoder_hidden_states_bart / decoder_hidden_states_bart.norm(dim=1, keepdim=True)

        scores = torch.matmul(decoder_hidden_states, decoder_hidden_states_bart.t())
        loss_bart_margin = loss_margin(
            scores.diag(),
            -torch.ones(decoder_hidden_states.shape[0]).to(decoder_hidden_states.device)
        )
        tr_margin_loss += loss_bart_margin.item()

   
        if not args.no_clip_loss:
            # Encode text
            image_emb = img_feat_cls
            text_emb = decode_and_encode_text(clip_tokenizer, model.clip_txt_model, tgt_ids_clip, DEVICE)
            similarity = image_emb @ text_emb.t()
            logits_per_image = similarity
            logits_per_text  = similarity.t()

            clip_gt = torch.arange(img_tensors.size(0), dtype=torch.long, device=DEVICE)
            total_loss_clip = (loss_img_clip(logits_per_image, clip_gt) + loss_txt_clip(logits_per_text, clip_gt)) / 2
            tr_clip_loss += total_loss_clip.item()

        face_name_loss = torch.tensor(0.0, device=DEVICE) 
        if not args.no_mapping:
            if args.use_secla:
            
                hidden_states_face = output["hidden_states_face"]
                hidden_states_names = get_embedding_ner(model=model, ner_ids_3d=names_ids_3d)
                hidden_states_names = hidden_states_names.to(DEVICE)

                face_name_loss = bi_contras_loss(hidden_states_face, hidden_states_names)
                tr_face_name_loss += face_name_loss.item()
            else:
                # Tính contrastive
                hidden_states_face = output["hidden_states_face"]
                hidden_states_face = pool(hidden_states_face, face_mask)
                hidden_states_face = hidden_states_face / hidden_states_face.norm(dim=1, keepdim=True)

                with torch.no_grad():
                    hidden_states_names = model(
                        input_ids=src_ids, attention_mask=src_mask,
                        decoder_input_ids=tgt_input,
                        image_features=img_feat_cls,
                        face_features=face_emb,
                        face_mask=face_mask,
                        name_ids=names_ids_flatten,
                        name_mask=names_cap_mask,
                        add_ner_ffn=False,
                        return_dict=True
                    )["hidden_states_ner"]

                hidden_states_names = hidden_states_names.to(DEVICE)
                hidden_states_names = pool(hidden_states_names, names_cap_mask.to(DEVICE))
                hidden_states_names = hidden_states_names / hidden_states_names.norm(dim=1, keepdim=True)

       
                scale = 1.0  

                logit_contras1 = scale * (hidden_states_names @ hidden_states_face.t())
                logit_contras2 = scale * (hidden_states_face @ hidden_states_names.t())

                clip_gt = torch.arange(img_tensors.size()[0], dtype=torch.long, device=DEVICE)
                face_name_loss = 0.5 * loss_clip_bart(logit_contras1, clip_gt) \
                               + 0.5 * loss_clip_bart(logit_contras2, clip_gt)
                tr_face_name_loss += face_name_loss.item()


        if not args.no_clip_loss:
            loss = total_loss_clip + txt_loss + args.mapping_loss_weight * face_name_loss + args.alpha * loss_bart_margin
        elif args.no_mapping:
            loss = txt_loss + args.alpha * loss_bart_margin
        else:
            loss = txt_loss + args.mapping_loss_weight * face_name_loss + args.alpha * loss_bart_margin

        # Backward
        loss.backward()
        if not args.no_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)

        tr_loss += loss.item()
        nb_tr_steps += 1

        optimizer_bart.step()
        scheduler_bart.step()
        optimizer_bart.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart, val_dataloader, DEVICE):
    model.eval()
    val_loss = 0
    nb_val_steps = 0
    out_dict = {}

    for step, batch in enumerate(tqdm(val_dataloader, desc="Iteration")):
        out_dict[step] = {}

        # Lấy batch
        src_ids, tgt_ids, tgt_sent, tgt_ids_clip, img_tensors, face_emb, names_art_ids = (
            batch["article_ids"],
            batch["caption_ids"],
            batch["caption"],
            batch["caption_ids_clip"],
            batch["img_tensor"],
            batch["face_emb"],
            batch["names_art_ids"],
        )

  
        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        tgt_ids_clip = tgt_ids_clip.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)

        # Tạo input decoder
        tgt_input = shift_tokens_right(tgt_ids, tokenizer.pad_token_id, tokenizer.eos_token_id)

        # Tạo attention mask
        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)

        # Gọi hàm extract_clip_img_feat dùng model.clip_img_model để encode ảnh

        img_feat, img_feat_cls = extract_clip_img_feat(model.clip_img_model, img_tensors)

        # Forward mô hình BART multi-modal
        if args.prompt_mlp_type == "clipcap":
            output = model(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=tgt_input,
                image_features=img_feat_cls,
                face_features=face_emb,
                face_mask=face_mask,
                name_ids=names_art_ids,
                name_mask=names_art_mask,
                add_ner_ffn=True,
                return_dict=True
            )
        else:
            output = model(
                input_ids=src_ids,
                attention_mask=src_mask,
                decoder_input_ids=tgt_input,
                image_features=img_feat,
                face_features=face_emb,
                face_mask=face_mask,
                name_ids=names_art_ids,
                name_mask=names_art_mask,
                add_ner_ffn=True,
                return_dict=True
            )

        # Tính loss cross-entropy
        logits = output["logits"]
        out_dict[step]["logit_output"] = [
            tokenizer.convert_tokens_to_string(
                tokenizer.convert_ids_to_tokens(torch.argmax(logits[i], dim=-1))
            )
            for i in range(logits.shape[0])
        ]
        txt_loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_ids.reshape(-1))
        loss = txt_loss

        out_dict[step]["gt_cap"] = tgt_sent

        # Nếu dùng DataParallel nhiều GPU, lấy trung bình
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        val_loss += loss.item()
        nb_val_steps += 1

    return val_loss / nb_val_steps, out_dict


def train(
    bart_model, model,
    loss_margin, loss_fn, loss_img_clip, loss_txt_clip, loss_clip_bart,
    train_dataloader, val_dataloader, test_dataloader,
    optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip,
    first_model, DEVICE
):
    train_losses = []
    val_losses = []
    min_val_loss = 999

    for epoch_i in range(int(1)): # args.num_epoch
        train_loss = train_epoch(
            bart_model, model,
            loss_margin, loss_fn,
            loss_img_clip, loss_txt_clip, loss_clip_bart,
            train_dataloader,
            optimizer_bart, optimizer_clip,
            scheduler_bart, scheduler_clip,
            epoch_i, DEVICE
        )

        val_loss, out_dict = eval_epoch(
            model, loss_fn,
            loss_img_clip, loss_txt_clip, loss_clip_bart,
            val_dataloader, DEVICE
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.eval()
            torch.save(model, os.path.join(args.out_dir, 'second_model' + ".pt"))
            with open(os.path.join(args.out_dir, 'second_model' + "v.json"), "w") as f:
                json.dump(out_dict, f)

        torch.save(model, os.path.join(args.out_dir, 'second_model' + "last.pt"))
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses

from transformers import GenerationConfig

def gen_caption_from_loader_bart(model, data_loader, tokenizer, bleu_scorer, rouge_scorer,
                                 cider_scorer, beam_size, max_length, DEVICE):
    rouge_scores = []
    count = 0
    out_dict = {}

    model.config.return_dict = True
    model.config.use_cache = False

    generation_config = GenerationConfig(
        num_beams=beam_size,
        max_length=max_length,
        return_dict_in_generate=True,
        use_cache=False, 
        output_scores=False, 
        output_attentions=False, 
        output_hidden_states=False  
    )

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
        out_dict[step] = {}
        src_ids, tgt_sent, img_tensors, face_emb, names_art_ids = (
            batch["article_ids"], batch["caption"], batch["img_tensor"],
            batch["face_emb"], batch["names_art_ids"]
        )
        src_ids = src_ids.to(DEVICE)
        img_tensors = img_tensors.to(DEVICE)
        face_emb = face_emb.to(DEVICE)
        names_art_ids = names_art_ids.to(DEVICE)
        src_mask = create_src_mask_bart(src_ids)
        face_mask = create_src_mask_bart(face_emb[:, :, -1])
        names_art_mask = create_src_mask_bart(names_art_ids)
        img_feat, img_feat_cls = extract_clip_img_feat(model.clip_img_model, img_tensors)

        # Sinh văn bản với mBART
        gen_output = model.generate(
            input_ids=src_ids,
            attention_mask=src_mask,
            generation_config=generation_config,  # Dùng GenerationConfig
            image_features=img_feat_cls if args.prompt_mlp_type == "clipcap" else img_feat,
            face_features=face_emb,
            face_mask=face_mask,
            name_ids=names_art_ids,
            name_mask=names_art_mask,
            add_ner_ffn=True
        )

    
        print(f"Step {step} - gen_output type: {type(gen_output)}")
        print(f"Step {step} - gen_output content: {gen_output}")

        if hasattr(gen_output, 'sequences'):
            gen_cap_ids = gen_output.sequences  # Trường hợp ModelOutput
        elif isinstance(gen_output, dict) and 'sequences' in gen_output:
            gen_cap_ids = gen_output['sequences']  # Trường hợp dictionary
        else:
            raise ValueError(f"Unexpected output format from model.generate(): {type(gen_output)}")

        gen_cap = tokenizer.batch_decode(gen_cap_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        gt_unidecode = unidecode.unidecode(tgt_sent[0])
        gen_unidecode = unidecode.unidecode(gen_cap)
        
        import re
        caption = re.sub(r'[^\w\s]', '', gt_unidecode)
        generation = re.sub(r'[^\w\s]', '', gen_unidecode)
        
        bleu_scorer += (generation, [caption])
        rouge_score = rouge_scorer.calc_score([generation], [caption])
        rouge_scores.append(rouge_score)
        cider_scorer += (generation, [caption])
        
        count += 1
        out_dict[step]["gt"] = gt_unidecode
        out_dict[step]["gen"] = gen_unidecode

    blue_score, _ = bleu_scorer.compute_score(option='closest')
    rouge_score = np.mean(np.array(rouge_scores))
    cider_score, _ = cider_scorer.compute_score()
    
    out_dict["bleu"] = {
        "bleu1": blue_score[0],
        "bleu2": blue_score[1],
        "bleu3": blue_score[2],
        "bleu4": blue_score[3]
    }
    out_dict["other metrics"] = {"rouge": rouge_score, "cider": cider_score}
    
    return out_dict, blue_score[0], blue_score[1], blue_score[2], blue_score[3], rouge_score, cider_score
def extract_visual_prompt(model, image_features, prompt_mlp_type):
    with torch.no_grad():
        image_features = model.model.prompt_mlp(image_features)
        if prompt_mlp_type == "clipcap":
            image_features = image_features.reshape(image_features.size()[0], args.prompt_size, 768)
        if model.model.embed_dim == 1024:
            image_features = model.model.visual_map(image_features)
    return image_features


def _stat(self, hypothesis_str, reference_list):
    hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
    score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
    score_line = score_line.replace('\n', '').replace('\r', '')
    self.meteor_p.stdin.write('{}\n'.format(score_line).encode())
    self.meteor_p.stdin.flush()
    return self.meteor_p.stdout.readline().decode().strip()


# Lớp tính loss phụ cho mapping
def batch_softmax(phrase_region_match):
    batch_size, _, num_spans, _ = phrase_region_match.size()
    phrase_region_max = phrase_region_match.max(-1).values
    phrase_region_scores = phrase_region_max.sum(-1)
    scale = torch.tensor(num_spans).expand(batch_size).unsqueeze(1).expand((batch_size, batch_size))
    scale = scale.to(phrase_region_scores.device)
    logits = phrase_region_scores.div(scale)
    targets = torch.arange(batch_size).to(logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)
        
class BatchSoftmax(torch.nn.Module):
    def __init__(self):
        super(BatchSoftmax, self).__init__()

    def forward(self, face_j, ner_j):
        face_ner_match = torch.matmul(ner_j.unsqueeze(1), face_j.permute(0, 2, 1))
        ner_face_match = torch.matmul(face_j.unsqueeze(1), ner_j.permute(0, 2, 1))
        loss1 = batch_softmax(face_ner_match)
        loss2 = batch_softmax(ner_face_match)
        loss = loss1 + loss2
        return loss



if __name__ == "__main__":
    import os
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    import torch
    import clip
    from venv import create
    from numpy import iterable
    import json
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup, BartTokenizer, PreTrainedTokenizerFast, BartForConditionalGeneration
    from models.modeling_mmbart_clip_inside_vis_clipcap_ent_type_final_fix_len_enc_self_face_name_ids_crossattn import MBartForMultiModalGeneration
    from torchvision import models, transforms
    from src.VNDataset import VNDataset,collate_fn_vndataset_entity_type
    import torch.optim as optim
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import re
    import types
    import numpy as np
    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    import unidecode
    import wandb
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.VNDataset import VNDataset, collate_fn_vndataset_entity_type
    from transformers import AutoTokenizer


    from models.modeling_mmbart_clip_inside_vis_clipcap_ent_type_final_fix_len_enc_self_face_name_ids_crossattn import MBartForMultiModalGeneration
    from torchvision import transforms
    plm_type_text = args.plm_type.replace("/", "-")
    clip_type_text = args.clip_type.replace("/", "-")
    ent_start = args.ent_start_token.replace("/", "-")
    ent_end = args.ent_end_token.replace("/", "-")

    # Đặt tên model dựa trên các tham số
    if args.only_image:
        if args.trained_clip == "no":
            first_model = f"m{args.margin}Match_{args.alpha}" + args.data_type + f"cross_newsmep_enc-self_retrieve{args.num_sentences}_FNID_secla{args.use_secla}_CLIPfreeze{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}"
        else:
            first_model = f"m{args.margin}Match_{args.alpha}" + args.data_type + f"cross_newsmep_enc-self_retrieve{args.num_sentences}_FNID_secla{args.use_secla}_trained_CLIPfreeze{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}"
    else:
        if args.trained_clip == "no":
            first_model = f"m{args.margin}Match_{args.alpha}" + args.data_type + f"cross_newsmep_enc-self_init{args.init_attn_weight}retri{args.num_sentences}_FNID_secla{args.use_secla}_CLIPfreeze{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}_bos{args.perturb}"
        else:
            first_model = f"m{args.margin}Match_{args.alpha}" + args.data_type + f"cross_newsmep_enc-self_init{args.init_attn_weight}_retri{args.num_sentences}_FNID_secla{args.use_secla}_trained_CLIPfreeze{clip_type_text}_{args.prompt_mlp_type}{args.prompt_size}{args.map_size}_{plm_type_text}_map{args.mapping_loss_weight}-{args.mapping_loss_type}_fuse{args.enc_fusion_layer}_dim{args.dim_common}_seed{args.seed}_bsz{args.train_batch_size}_lr{args.lr_bart}-{args.clip_norm}_{args.num_epoch}epoch_warm{args.warmup_rate}{args.weight_decay}_len{args.article_max_length}_bos{args.perturb}"
    if not args.freeze_clip:
        first_model = first_model.replace("freeze", "")
    if args.prompt_mlp_type == "clipcap":
        first_model = first_model.replace(f"{args.map_size}", "")
    elif args.prompt_mlp_type == "mlp":
        first_model = first_model.replace(f"{args.prompt_size}", "")
    if args.enc_fusion_layer == [0,1,2,3,4,5] and args.dim_common == 768:
        first_model = first_model.replace(f"{args.enc_fusion_layer}", "all-enc")
    elif args.enc_fusion_layer == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] and args.dim_common == 1024:
        first_model = first_model.replace(f"{args.enc_fusion_layer}", "all-enc")
    elif args.enc_fusion_layer == [0, 1, 2, 3, 4, 5] and args.dim_common == 1024:
        first_model = first_model.replace(f"{args.enc_fusion_layer}", "front-enc")
    elif args.enc_fusion_layer == [6, 7, 8, 9, 10, 11] and args.dim_common == 1024:
        first_model = first_model.replace(f"{args.enc_fusion_layer}", "back-enc")
    if args.no_mapping:
        first_model = first_model.replace(f"_map{args.mapping_loss_weight}-{args.mapping_loss_type}", "nomap")
    if args.no_clip_norm:
        first_model = first_model.replace(f"-{args.clip_norm}", "")
    first_model = first_model.replace("True", "T")
    first_model = first_model.replace("False", "F")
    first_model = first_model.replace("patrickvonplaten", "")

    if args.plm_type.startswith("ainize"):
        tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_type,src_lang="vi_VN", tgt_lang="vi_VN")
    # Nếu không dùng CLIP đã được huấn luyện sẵn
    if args.trained_clip == "no":
        from sentence_transformers import SentenceTransformer
        img_model = SentenceTransformer('/data/npl/ICEK/VACNIC/src/data/assest/clip-ViT-B-32')
        text_model = SentenceTransformer('/data/npl/ICEK/VACNIC/src/data/assest/clip-ViT-B-32-multilingual-v1')
        clip_tokenizer = text_model.tokenizer  
        # clip_model, clip_preprocess = clip.load(args.clip_type, device=DEVICE)
    else:
        clip_model = torch.load(os.path.join(args.clip_dir, args.trained_clip), map_location=torch.device('cpu'))
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # Khởi tạo model
    from models.modeling_mmbart_clip_inside_vis_clipcap_ent_type_final_fix_len_enc_self_face_name_ids_crossattn import MBartForMultiModalGeneration
    model = MBartForMultiModalGeneration.from_pretrained(
        args.plm_type,
        output_hidden_states=True,
        enc_fusion_layer=args.enc_fusion_layer,
        dim_common=args.dim_common,
        img_size=args.img_size,
        prompt_mlp_type=args.prompt_mlp_type,
        map_size=args.map_size,
        prompt_size=args.prompt_size,
        clip_img_model=img_model,  
        clip_txt_model=text_model,
        freeze_clip=args.freeze_clip,
        max_ner_type_len=args.max_ner_type_len,
        max_ner_type_len_gt=args.max_ner_type_len_gt,
        only_image=args.only_image,
        init_attn_weight=args.init_attn_weight
    )
    from transformers import MBartTokenizer, MBartForConditionalGeneration
    # bart_model = AutoModelForSeq2SeqLM.from_pretrained(args.plm_type, trust_remote_code=True)
    from transformers import AutoModelForSeq2SeqLM
    bart_model = AutoModelForSeq2SeqLM.from_pretrained(args.plm_type, trust_remote_code=True)

    bart_model = bart_model.to(DEVICE)
    for param in bart_model.parameters():
        param.requires_grad = False
    tokenizer.add_special_tokens({"additional_special_tokens": ['<ENT>', "<NONAME>"]})
    model.resize_token_embeddings(len(tokenizer))
    if args.perturb:
        bos_noise = torch.randn(1024)
        model.model.shared.weight.data[0] = model.model.shared.weight.data[0] + bos_noise
    del img_model
    del text_model

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer_dataset = AutoTokenizer.from_pretrained(args.plm_type, trust_remote_code=True)
    # tokenizer_dataset = BartTokenizer.from_pretrained(args.plm_type)
    tokenizer_dataset.add_special_tokens({"additional_special_tokens":['<ENT>', "<NONAME>", '<PERSON>', "<ORGNORP>", "<GPELOC>"]})
    person_token_id = tokenizer_dataset.convert_tokens_to_ids("<PERSON>")
    # Xử lý dữ liệu dựa trên args.data_type
    if args.data_type == "nytimes":
        pass
    elif args.data_type == "goodnews":
        pass
    elif args.data_type == "vietnamese":
        data_base_dir = '/data/npl/ICEK/VACNIC/data/train'
        with open('/data/npl/ICEK/VACNIC/data/train/test2.json','r',encoding='utf-8') as f:
            train_dict = json.load(f)
        train_data = VNDataset(train_dict, data_base_dir, tokenizer_dataset,
                               use_clip_tokenizer=True,
                               entity_token_start=args.ent_start_token,
                               entity_token_end=args.ent_end_token,
                               transform=img_transform,
                               max_article_len=args.article_max_length,
                               max_ner_type_len=args.max_ner_type_len,
                               max_ner_type_len_gt=args.max_ner_type_len_gt,
                               retrieved_sent=True,
                               person_token_id=person_token_id)
        train_loader = DataLoader(train_data, args.train_batch_size, num_workers=args.num_workers,
                                  collate_fn=collate_fn_vndataset_entity_type)
        with open('/data/npl/ICEK/VACNIC/data/train/test3.json','r',encoding='utf-8') as f:
            val_dict = json.load(f)
        val_data = VNDataset(val_dict, data_base_dir, tokenizer_dataset,
                             use_clip_tokenizer=True,
                             entity_token_start=args.ent_start_token,
                             entity_token_end=args.ent_end_token,
                             transform=img_transform,
                             max_article_len=args.article_max_length,
                             max_ner_type_len=args.max_ner_type_len,
                             max_ner_type_len_gt=args.max_ner_type_len_gt,
                             retrieved_sent=True,
                             person_token_id=person_token_id)
        val_loader = DataLoader(val_data, args.val_batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn_vndataset_entity_type)
        with open('/data/npl/ICEK/VACNIC/data/train/test1.json','r',encoding='utf-8') as f:
            test_dict = json.load(f)
        test_data = VNDataset(test_dict, data_base_dir, tokenizer_dataset,
                              use_clip_tokenizer=True,
                              entity_token_start=args.ent_start_token,
                              entity_token_end=args.ent_end_token,
                              transform=img_transform,
                              max_article_len=args.article_max_length,
                              max_ner_type_len=args.max_ner_type_len,
                              max_ner_type_len_gt=args.max_ner_type_len_gt,
                              retrieved_sent=True,
                              person_token_id=person_token_id)
        test_loader = DataLoader(test_data, args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                 collate_fn=collate_fn_vndataset_entity_type)
    else:
        train_loader = None
        val_loader = None
        test_loader = None

    model, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip = prep_for_training(model, len(train_data), DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(DEVICE)
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    loss_clip_bart = torch.nn.CrossEntropyLoss()
    loss_margin = torch.nn.HingeEmbeddingLoss(margin=args.margin)
    train(bart_model, model, loss_margin, loss_fn, loss_img, loss_txt, loss_clip_bart,
          train_loader, val_loader, test_loader, optimizer_bart, optimizer_clip, scheduler_bart, scheduler_clip,
          first_model, DEVICE)
    

    from pycocoevalcap.bleu.bleu_scorer import BleuScorer
    from pycocoevalcap.cider.cider_scorer import CiderScorer
    from pycocoevalcap.rouge.rouge import Rouge

    bleu_scorer = BleuScorer(n=4)
    rouge_scorer = Rouge()
    cider_scorer = CiderScorer(n=4, sigma=6.0)


    test_out_dict, blue1, blue2, blue3, blue4, rouge_score, cider_score = \
    gen_caption_from_loader_bart(model, test_loader, tokenizer, bleu_scorer, rouge_scorer,
                                  cider_scorer, args.beam_size, args.max_length, DEVICE)
    with open(os.path.join(args.out_dir, 'second_model' + "last.json"), "w") as f:
        json.dump(test_out_dict, f)
    tokenizer.save_pretrained(args.out_dir)



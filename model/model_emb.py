DB_ID_LIST = ['activity_1', 'party_people', 'phone_market', 'tracking_orders', 'college_1', 'student_1', 'manufactory_1', 'book_2', 'election', 'small_bank_1', 'student_assessment', 'machine_repair', 'csu_1', 'shop_membership', 'local_govt_and_lot', 'voter_1', 'hospital_1', 'decoration_competition', 'network_1', 'cre_Theme_park', 'loan_1', 'tracking_software_problems', 'local_govt_mdm', 'insurance_fnol', 'train_station', 'customers_card_transactions', 'company_1', 'soccer_1', 'city_record', 'flight_1', 'flight_2', 'course_teach', 'climbing', 'body_builder', 'ship_mission', 'customer_complaints', 'sakila_1', 'department_store', 'world_1', 'network_2', 'poker_player', 'culture_company', 'store_product', 'gymnast', 'assets_maintenance', 'apartment_rentals', 'products_for_hire', 'movie_1', 'theme_gallery', 'solvency_ii', 'customers_and_products_contacts', 'musical', 'bike_1', 'car_1', 'college_3', 'voter_2', 'school_bus', 'game_1', 'cre_Doc_Template_Mgt', 'game_injury', 'baseball_1', 'aircraft', 'soccer_2', 'inn_1', 'pilot_record', 'debate', 'customers_and_invoices', 'riding_club', 'mountain_photos', 'architecture', 'club_1', 'wrestler', 'concert_singer', 'farm', 'employee_hire_evaluation', 'race_track', 'battle_death', 'match_season', 'station_weather', 'real_estate_properties', 'browser_web', 'school_player', 'phone_1', 'device', 'insurance_policies', 'wine_1', 'music_1', 'flight_company', 'behavior_monitoring', 'election_representative', 'protein_institute', 'tvshow', 'journal_committee', 'tracking_grants_for_research', 'icfp_1', 'museum_visit', 'customer_deliveries', 'insurance_and_eClaims', 'storm_record', 'perpetrator', 'e_government', 'product_catalog', 'singer', 'music_2', 'allergy_1', 'sports_competition', 'entrepreneur', 'school_finance', 'driving_school', 'music_4', 'college_2', 'e_learning', 'swimming', 'flight_4', 'university_basketball', 'entertainment_awards', 'candidate_poll', 'county_public_safety', 'coffee_shop', 'restaurant_1', 'company_office', 'railway', 'program_share', 'tracking_share_transactions', 'cre_Drama_Workshop_Groups', 'cre_Doc_Tracking_DB', 'epinions_1', 'chinook_1', 'wedding', 'orchestra', 'cre_Docs_and_Epenses', 'pets_1', 'news_report', 'store_1', 'twitter_1', 'products_gen_characteristics', 'dog_kennels', 'hr_1', 'film_rank', 'student_transcripts_tracking', 'workshop_paper', 'customers_campaigns_ecommerce', 'customers_and_addresses', 'party_host', 'manufacturer', 'company_employee', 'document_management', 'medicine_enzyme_interaction', 'cre_Doc_Control_Systems', 'performance_attendance', 'gas_company', 'formula_1', 'cinema', 'ship_1', 'dorm_1', 'roller_coaster', 'scientist_1', 'department_management', 'local_govt_in_alabama']
'''
model with provided pre-generated table embeddings
need a db_path in config and a db_id in each input to retrieve the table embeddings
'''
from model.decoder import TableDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Sequence, List, Tuple, Union
import json, os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MyConfig(PretrainedConfig):
    model_type = "my_emb_model"

    def __init__(self, device='cuda', mm_projector_type='linear', encoder_hidden_size=384,
                 decoder_hidden_size=4096, base_model_name_or_path=None, model_path=None, db_path=None, torch_dtype="float32"):
        super().__init__(torch_dtype=torch_dtype) 
        self.device = device
        self.mm_projector_type = mm_projector_type
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.base_model_name_or_path = base_model_name_or_path
        self.model_path = model_path
        self.db_path = db_path
        
    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None, **kwargs): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; otherwise a new model will be created with the provided config_dict
        if model_path is not None:
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config_dict = json.load(f)
        
        elif config_dict is None:
            raise ValueError("The 'config_dict' must be provided if 'model_path' is None.")            
        
        base_model_name_or_path = config_dict.get('base_model_name_or_path')
        if base_model_name_or_path is None:
            raise ValueError("The config file must contain the 'base_model_name_or_path' field.")
        
        # base_config = AutoConfig.from_pretrained(base_model_name_or_path)
        
        all_dict = {
            **config_dict,
            **kwargs,
            'model_path': model_path
        }
        return cls(**all_dict)
    
class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, config, decoder, tokenizer):
        super().__init__()
        self.config = config
        self.gradient_checkpointing_enable = decoder.model.gradient_checkpointing_enable
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.torch_dtype = config.torch_dtype
        self.db_path = config.db_path

    @classmethod
    def from_pretrained(cls, model_path=None, config_dict = None): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; else a new model will be created with the provided config_dict
        config = MyConfig.from_pretrained(model_path=model_path, config_dict=config_dict)
        
        decoder = TableDecoder.from_pretrained(config=config)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        import os
        model = cls(config, decoder, tokenizer = tokenizer)
        return model
        
    def get_db_embeds(self, db_id, table=None):
        # db_id = np.random.choice(DB_ID_LIST) # randomly select a db_id from DB_ID_LIST
        
        db_path = self.config.db_path
        cur_db_path = os.path.join(db_path, db_id)
        assert (table != None)
        if table == None:
            npy_list = [os.path.join(cur_db_path, file) for file in os.listdir(cur_db_path) if file.endswith('.npy')]
            ret = torch.cat([torch.tensor(np.load(os.path.join(cur_db_path, npy_file)), dtype=self.torch_dtype, device=self.decoder.model.device, requires_grad=False).detach() for npy_file in npy_list], dim=0)
        else:
            npy_file = table + '.npy'
            assert os.path.exists(os.path.join(cur_db_path, npy_file))
            ret = torch.tensor(np.load(os.path.join(cur_db_path, npy_file)), dtype=self.torch_dtype, device=self.decoder.model.device, requires_grad=False).detach()
        assert len(ret.shape) == 2, "len(ret.shape) != 2"
        ret = F.normalize(ret)
        return ret
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        db_ids = None,
        tables = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]: # ?
        bs = input_ids.shape[0]
        table_embeds = []
        for i in range(bs):
            table_embeds.append(self.get_db_embeds(db_ids[i], tables[i]))
        # TODO: 防止多次加载相同的表格。似乎表格长度都有限，感觉可以强行存
        
        return self.decoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            table_embeds=table_embeds
        )
    
    def generate(self, input_str = None, input_ids = None, db_id = None, table = None, max_new_tokens = 114514, **kwargs):
        if input_ids is None:
            input_ids = self.tokenizer(input_str, return_tensors='pt')['input_ids'].to(self.decoder.model.device)
        bs = input_ids.shape[0]
        assert bs == 1, "batch size must be 1"
        
        table_embeds = self.get_db_embeds(db_id, table).unsqueeze(0)
        assert len(table_embeds.shape) == 3, "table_embeds must be 3D" # bs * n_tokens * hidden_size
        # extended_table_embeds = self.learnable_embedding.repeat(bs, 1, 1)
        return self.decoder.generate(input_ids, max_new_tokens, table_embeds, **kwargs)
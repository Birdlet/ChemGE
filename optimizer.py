import nltk
import numpy as np
import utilities as util
import scorer
from config import GCFG


class optimizer():
    def __init__(self):
        self._population = []
        self._grammer = "zinc_grammer"
        self._GCFG = GCFG
        self._productions = self._GCFG.productions()
        self._productions_map  = self._get_productions_map()
        self._tokenize = self._get_zinc_tokenizer()
        self._parser = nltk.ChartParser(self._GCFG)

    def _get_zinc_tokenizer(self):
        # Return tokenizer for smiles
        long_tokens = [a for a in self._GCFG._lexical_index.keys() if len(a) > 1]
        replacements = ['$', '%', '^']
        assert len(long_tokens) == len(replacements)
        for token in replacements:
            assert token not in self._GCFG._lexical_index

        def tokenize(smiles):
            """Tokenize smiles, called by `self.encode` function
            #
            # Input:
            #   smiles : smiles to be tokenized
            # Return:
            #   a list of char with tokenized smiles 
            #
            # Example:
            #   "O=C=O" -> ['O', '=', 'C', 'O', '=']
            """
            for i, token in enumerate(long_tokens):
                smiles = smiles.replace(token, replacements[i])
            tokens = []
            for token in smiles:
                try:
                    ix = replacements.index(token)
                    tokens.append(long_tokens[ix])
                except Exception:
                    tokens.append(token)
            return tokens
        return tokenize


    def _get_productions_map(self):
        # Return a dict of production map
        productions_map = {}
        for ix, prod in enumerate(self._productions):
            productions_map[prod] = ix
        return productions_map


    def decode(self, gene):
        """Generate productions of smiles
        #
        # Input:
        #   gene : np.ndarray of int
        #           smiles to be encoded
        # Return:
        #   smiles: string
        #   return smiles if it is a valid molecular, or return ``""`` empty string
        #
        # Example:
        #   array([ 0, 72, 72, 70, 60,  2,  5, 54, 60,  2,  7, 54, 60,  2,  5]) -> "O=C=O"
        """
        rule = self.gene_to_rules(gene)
        prod_seq = [self._productions[i] for i in rule]

        # prod_to_eq
        seq = [prod_seq[0].lhs()]
        for prod in prod_seq:
            if str(prod.lhs()) == 'Nothing':
                break
            for ix, s in enumerate(seq):
                if s == prod.lhs():
                    seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                    break
        try:
            smiles = ''.join(seq)
        except Exception:
            smiles = ''
        return smiles


    def encode(self, smiles, max_len=-1):
        """Encode smiles to gene
        #
        # Input:
        #   smiles : string
        #           smiles to be encoded
        #   max_len : int
        #           maxium gene length allowed, '-1' means unlimited,
        #           default is ``-1``.
        # Return:
        #   gene : np.ndarray of int
        #   procution rules for smiles  
        #
        # Example:
        #   "O=C=O" -> array([ 0, 72, 72, 70, 60,  2,  5, 54, 60,  2,  7, 54, 60,  2,  5])
        """
        production_rules = self.smiles_to_rules(smiles)
        gene = []
        for r in production_rules:
            lhs = self._productions[r].lhs()
            possible_rules = [idx for idx, rule in enumerate(self._productions)
                            if rule.lhs() == lhs]
            gene.append(possible_rules.index(r))
        if max_len > 0:
            if len(gene) > max_len:
                gene = gene[:max_len]
            else:
                gene = gene + [np.random.randint(0, 256)
                            for _ in range(max_len-len(gene))]
        return gene


    def smiles_to_rules(self, smiles):
        """Generate productions of smiles
        #
        # Input:
        #   smiles : string
        #           smiles to be encoded; this function should only be called
        #           by ``self._encode``
        # Return:
        #   indices : np.ndarray of int
        #           procution rules for smiles, a np.ndarray of int  
        #
        # Example:
        #   "O=C=O" -> array([ 0, 72, 72, 70, 60,  2,  5, 54, 60,  2,  7, 54, 60,  2,  5])
        """
        # tokenize smiles
        tokens = self._tokenize(smiles)
        # return the only one nltk parse tree
        parse_tree = next(self._parser.parse(tokens)) 
        productions_of_smiles= parse_tree.productions()
        indices = np.array([self._productions_map[prod] for prod in productions_of_smiles],
            dtype=int)
        return indices


    def gene_to_rules(self, gene):
        """Reture productions of gene
        #
        # Input:
        #   gene: gene for a smiles, list of int
        # Return:
        #   procution rules for gene, a list of int
        #
        # smiles "O=C=O":
        # [0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 3, 1, 0, 1, 1] -> 
        #    [0, 72, 72, 70, 60, 2, 5, 54, 60, 2, 7, 54, 60, 2, 5]
        """
        production_rules = []
        stack = [self._productions[0].lhs()]
        for g in gene:
            try:
                lhs = stack.pop()
            except Exception:
                break
            possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                            if rule.lhs() == lhs]
            rule = possible_rules[g % len(possible_rules)]
            production_rules.append(rule)
            rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal)
                        and (str(a) != 'None'),
                        self._GCFG.productions()[rule].rhs())
            stack.extend(list(rhs)[::-1])
        return production_rules
    
    
    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)


    def optimize(self, smiles, mu, lam, generation, scorer, log=None, seed=-1, verbose=True):
        """Run optimizer for a given population with a target
        #
        # Input:
        #   gene: gene for a smiles, list of int
        # Return:
        #   procution rules for gene, a list of int
        #
        # smiles "O=C=O":
        # [0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 3, 1, 0, 1, 1] -> 
        #    [0, 72, 72, 70, 60, 2, 5, 54, 60, 2, 7, 54, 60, 2, 5]
        """
        if seed > -1:
            np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(smi) for smi in initial_smiles]
        initial_genes = [self.encode(s, max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = []
        print(r"|0%--------------------50%-------------------100%|")
        for i, smi in enumerate(initial_smiles):
            initial_scores.append(scorer(smi))
            print("*"*int(50*i/(mu+lam)), end='\r')
        print("\nInitialize finished!")

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)
                c_smiles = util.canonicalize(self.decode(c_gene))
                #print(c_smiles, c_smiles not in all_smiles)
                # Umm, not as good as I supposed
                if c_smiles not in all_smiles:
                    c_score = scorer(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size =len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result


class optimizerJ(optimizer):
    def __init__(self):
        super().__init__()


    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w+") as f:
            f.write("score,smiles\n")
            for data in results:
                line = "{},{}\n".format(data[0], data[1])
                f.write(line)


    def optimize(self, smiles, mu=32, lam=64, generation=1000, log=None, seed=-1, verbose=True):
        super().optimize(smiles, mu, lam, generation,
            scorer=scorer.scorej, log=log, seed=seed, verbose=verbose)


class optimizerRDock(optimizer):
    def __init__(self):
        optimizer.__init__(self)

    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)



    def optimize(self, smiles, log = None, mu=32, lam=64, generation=1000, seed=0, verbose=True):
        np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(s) for s in initial_smiles]
        initial_genes = [self.encode(s, max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = [scorer.score(s) for s in initial_smiles]

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)

                c_smiles = util.canonicalize(self.decode(c_gene))
                if c_smiles not in all_smiles:
                    c_score = scorer.score(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size =len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result



class optimizerVina(optimizer):
    def __init__(self):
        optimizer.__init__(self)

    def _log(self, epoches, population, population_size):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]

        print("Generation:{:<4d},{:8.2f},{:8.2f},  {},{:4d}"
            .format(epoches, mean_score, best_score, best_smiles, population_size))
            #file = sys.stderr)


    def _log_file(self, log, results):
        with open(log, "w") as f:
            for data in results:
                line = "{:12.4f},{}\n".format(data[0], data[1])
                f.write(line)



    def optimize(self, smiles, vina_conf, vina_log_path, \
        log = None, mu=32, lam=64, generation=1000, seed=0, verbose=True):

        np.random.seed(seed)
        gene_length = 300

        # Initialize population
        print("Initializing Population....")

        # Generation 0, start from input smiles
        initial_smiles = np.random.choice(smiles, mu+lam) 
        initial_smiles = [util.canonicalize(s) for s in initial_smiles]
        initial_genes = [self.encode(s, max_len=gene_length)
                        for s in initial_smiles]
        initial_scores = []
        print(r"|0%--------------------50%-------------------100%|")
        for i, s in enumerate(initial_smiles):
            scorer.score_vina(s, conf_path=vina_conf, log_path=vina_log_path)
            print("*"*int(50*i/(mu+lam)), end='\r')
        print()

        population = []
        for score, gene, smiles in zip(initial_scores, initial_genes,
                                    initial_smiles):
            population.append((score, smiles, gene))

        # Select top $mu$ smiles as generation 0
        population = sorted(population, key=lambda x: x[0], reverse=True)[:mu]
    

        # Start!
        print("Generation Start!")
        all_smiles = [p[1] for p in population]
        all_result = []

        for epoch in range(generation):
            
            new_population = []
            # For each mutation in each generation in range $lamda$
            for _ in range(lam):
                # random select one smi/gene in top $mu$ smiles
                p = population[np.random.randint(mu)] 
                p_gene = p[2]
                c_gene = util.mutation(p_gene)

                c_smiles = util.canonicalize(self.decode(c_gene))
                if c_smiles not in all_smiles:
                    c_score = scorer.score(c_smiles)
                    c = (c_score, c_smiles, c_gene)
                    new_population.append(c)
                    all_smiles.append(c_smiles)

            population.extend(new_population)
            all_result.extend(new_population)
            population = sorted(population,
                                key=lambda x: x[0], reverse=True)[:mu]

            if epoch%15 == 0 and verbose:
                # Log on screen
                self._log(epoch, population, population_size=len(all_smiles))

        print("\nFinished!")

        if log:
            try:
                self._log_file(log, all_result)
                print("Log file write into %s" % log)
            except:
                print("Failed writing log to %s" % log)
        
        return all_result

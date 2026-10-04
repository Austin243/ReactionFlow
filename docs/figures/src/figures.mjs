// Every figure, in gallery order.

import bandAnimation from './figures/band-animation.mjs';
import campaign from './figures/campaign.mjs';
import campaigns from './figures/campaigns.mjs';
import detection from './figures/detection.mjs';
import eonLandscape from './figures/eon-landscape.mjs';
import eonLoop from './figures/eon-loop.mjs';
import eonMode from './figures/eon-mode.mjs';
import eonOutcomes from './figures/eon-outcomes.mjs';
import eonRunDirectory from './figures/eon-run-directory.mjs';
import eonSearch from './figures/eon-search.mjs';
import eonStep from './figures/eon-step.mjs';
import ensembles from './figures/ensembles.mjs';
import hero, { heroAnimated } from './figures/hero.mjs';
import howItWorks from './figures/how-it-works.mjs';
import identity from './figures/identity.mjs';
import mdMode from './figures/md-mode.mjs';
import models from './figures/models.mjs';
import outcomes from './figures/outcomes.mjs';
import overview from './figures/overview.mjs';
import pathway from './figures/pathway.mjs';
import phases from './figures/phases.mjs';
import restart from './figures/restart.mjs';
import runDirectory from './figures/run-directory.mjs';

export const FIGURES = [
  overview, mdMode, eonMode, campaigns,
  hero, howItWorks, detection, identity, pathway, ensembles, outcomes, restart, campaign, models, runDirectory, phases,
  eonLandscape, eonLoop, eonSearch, eonOutcomes, eonStep, eonRunDirectory,
  heroAnimated, bandAnimation,
];

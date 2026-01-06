#!/usr/bin/env node

const { Command } = require('commander');
const fs = require('fs');
const { AgentOrchestrator } = require('./dist/index.js');

const program = new Command();

program
  .name('agent-orchestrate')
  .description('CLI for multi-agent orchestration')
  .version('1.0.0')
  .requiredOption('-s, --spec <path>', 'path to the YAML spec file')
  .option('-v, --verbose', 'enable verbose logging')
  .action((options) => {
    const specPath = options.spec;
    
    if (!fs.existsSync(specPath)) {
      console.error(`Spec file does not exist: ${specPath}`);
      process.exit(1);
    }
    
    const specContent = fs.readFileSync(specPath, 'utf8');
    const orchestrator = new AgentOrchestrator();
    
    orchestrator.execute(specContent, {
      verbose: options.verbose
    }).catch(err => {
      console.error('Error executing orchestration:', err);
      process.exit(1);
    });
  });

program.parse();